import os
import shutil
import string
import sys

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import flopy
import pyemu
sys.path.append("..")
import processing
from string import ascii_uppercase

abet = string.ascii_uppercase

unit_dict = {"head":"sw-gw flux $\\frac{ft^3}{d}$",
                "tail": "sw-gw flux $\\frac{ft^3}{d}$",
                "trgw" : "gw level $ft$",
                "gage" : "sw flux $\\frac{ft^3}{d}$"}
label_dict = {"head": "headwater",
             "tail": "tailwater",
             "trgw_2_29_5": "gw_1",
              "trgw_2_101_23": "gw_2",
             "gage": "sw_1"}

if sys.platform.startswith('win'):
    exe_dir = os.path.join("..","..","bin","win")
    mf_exe = 'mf6.exe'
    pst_exe = 'pestpp-ies.exe'

elif sys.platform.startswith('linux'):
    exe_dir = os.path.join("..","..","bin","linux")

    mf_exe = 'mf6'
    pst_exe = 'pestpp-ies'

    os.system(f'chmod +x {os.path.join(exe_dir, mf_exe)}')
    os.system(f'chmod +x {os.path.join(exe_dir, pst_exe)}')

elif sys.platform.lower().startswith('dar') or sys.platform.lower().startswith('mac'):
    exe_dir = os.path.join("..","..","bin","mac")
    mf_exe = 'mf6'
    pst_exe = 'pestpp-ies'

    os.system(f'chmod +x {os.path.join(exe_dir, mf_exe)}')
    os.system(f'chmod +x {os.path.join(exe_dir, pst_exe)}')
else:
    raise Exception('***ERROR: OPERATING SYSTEM UNKOWN***')


def setup_interface(org_ws,num_reals=100,full_interface=True,include_constants=True):
    # load the mf6 model with flopy to get the spatial reference
    sim = flopy.mf6.MFSimulation.load(sim_ws=org_ws)
    m = sim.get_model("freyberg6")

    #fix the fucking wrapped format bullshit from the 1980s
    sim.simulation_data.max_columns_of_data = m.modelgrid.ncol

    # work out the spatial rediscretization factor
    redis_fac = m.dis.nrow.data / 40


    # where the pest interface will be constructed
    template_ws = org_ws.split('_')[1] + "_template"
    if not full_interface:
        template_ws += "_cond"
    
    
    # instantiate PstFrom object
    pf = pyemu.utils.PstFrom(original_d=org_ws, new_d=template_ws,
                remove_existing=True,
                longnames=True, spatial_reference=m.modelgrid,
                zero_based=False,start_datetime="1-1-2018")

    # the geostruct object for grid-scale parameters
    grid_v = pyemu.geostats.ExpVario(contribution=1.0,a=500)
    grid_gs = pyemu.geostats.GeoStruct(variograms=grid_v)

    # the geostruct object for pilot-point-scale parameters
    pp_v = pyemu.geostats.ExpVario(contribution=1.0, a=3000)
    pp_gs = pyemu.geostats.GeoStruct(variograms=pp_v)

    # the geostruct for recharge grid-scale parameters
    rch_v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
    rch_gs = pyemu.geostats.GeoStruct(variograms=rch_v)

    # the geostruct for temporal correlation
    temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0,a=60))
    
    # import flopy as part of the forward run process
    pf.extra_py_imports.append('flopy')

    # use the idomain array for masking parameter locations
    ib = m.dis.idomain[0].array
    
    # define a dict that contains file name tags and lower/upper bound information
    #tags = {"npf_k_":[0.1,10.],"npf_k33_":[.1,10],"sto_ss":[.1,10],"sto_sy":[.9,1.1]}#,
            #"rch_recharge":[.5,1.5]}
    tags = {"npf_k_": [0.1, 10.], "sto_ss": [.1, 10]}
    dts = pd.to_datetime("1-1-2018") + \
          pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]),unit="d")

    # loop over each tag, bound info pair
    for tag,bnd in tags.items():
        if not full_interface and "_k_" not in tag and "_ss" not in tag and "_sy" not in tag and "k33" not in tag:
            continue

        lb,ub = bnd[0],bnd[1]
        # find all array based files that have the tag in the name
        arr_files = [f for f in os.listdir(template_ws) if tag in f and f.endswith(".txt")]

        if len(arr_files) == 0:
            print("warning: no array files found for ",tag)
            continue
        
        # make sure each array file in nrow X ncol dimensions (not wrapped)
        for arr_file in arr_files:
            arr = np.loadtxt(os.path.join(template_ws,arr_file)).reshape(ib.shape)
            np.savetxt(os.path.join(template_ws,arr_file),arr,fmt="%15.6E")
        
        # if this is the recharge tag
        # if "rch" in tag:
        #     # add one set of grid-scale parameters for all files
        #     pf.add_parameters(filenames=arr_files, par_type="grid", par_name_base="rch_gr",
        #                       pargp="rch_gr", zone_array=ib, upper_bound=ub, lower_bound=lb,
        #                       geostruct=rch_gs)
        #
        #     # add one constant parameter for each array, and assign it a datetime
        #     # so we can work out the temporal correlation
        #     for arr_file in arr_files:
        #         kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
        #         pf.add_parameters(filenames=arr_file,par_type="constant",par_name_base=arr_file.split('.')[1]+"_cn",
        #                           pargp="rch_const",zone_array=ib,upper_bound=ub,lower_bound=lb,geostruct=temporal_gs,
        #                           datetime=dts[kper])
        # otherwise...
        else:
            # for each array add both grid-scale and pilot-point scale parameters
            for arr_file in arr_files:
                if ("sy" in arr_file and
                        int(arr_file.strip(".txt").split('layer')[-1]) > 1):
                    continue
                pf.add_parameters(filenames=arr_file,par_type="grid",par_name_base=arr_file.split('.')[1].replace("_","")+"_gr",
                                  pargp=arr_file.split('.')[1].replace("_","")+"_gr",zone_array=ib,upper_bound=ub,lower_bound=lb,
                                  geostruct=grid_gs)
                pf.add_parameters(filenames=arr_file, par_type="pilotpoints", par_name_base=arr_file.split('.')[1].replace("_","")+"_pp",
                                  pargp=arr_file.split('.')[1].replace("_","")+"_pp", zone_array=ib,upper_bound=ub,lower_bound=lb,
                                  pp_space=int(5 * redis_fac),geostruct=pp_gs)
                if include_constants:
                    pf.add_parameters(filenames=arr_file, par_type="constant",
                                      par_name_base=arr_file.split('.')[1].replace("_", "") + "_cn",
                                      pargp=arr_file.split('.')[1].replace("_", "") + "_cn", zone_array=ib,
                                      upper_bound=ub, lower_bound=lb)

                ar = np.loadtxt(os.path.join(pf.new_d, arr_file))
                np.savetxt(os.path.join(pf.new_d, f"log_{arr_file}"),
                           np.log10(ar))
                np.savetxt(os.path.join(pf.new_d, f"log_dup_{arr_file}"),
                           np.log10(ar))
                pf.add_observations(filename=f"log_{arr_file}",
                                    obsgp=arr_file.split('.')[1].replace("_", ""),
                                    prefix=arr_file.split('.')[1].replace("_", ""))
                if "_ss" in tag or "_sy" in tag:
                    pf.add_observations(filename=f"log_dup_{arr_file}",
                                        obsgp="dup-"+arr_file.split('.')[1].replace("_", ""),
                                        prefix="dup-"+arr_file.split('.')[1].replace("_", ""))


    if full_interface:
        # # get all the list-type files associated with the wel package
        # list_files = [f for f in os.listdir(org_ws) if "freyberg6.wel_stress_period_data_" in f and f.endswith(".txt")]
        # # for each wel-package list-type file
        # for list_file in list_files:
        #     kper = int(list_file.split(".")[1].split('_')[-1]) - 1
        #     # add spatially constant, but temporally correlated parameter
        #     pf.add_parameters(filenames=list_file,par_type="constant",par_name_base="twel_mlt_{0}".format(kper),
        #                       pargp="twel_mlt".format(kper),index_cols=[0,1,2],use_cols=[3],
        #                       upper_bound=1.5,lower_bound=0.5, datetime=dts[kper], geostruct=temporal_gs)
        #
        #     # add temporally indep, but spatially correlated grid-scale parameters, one per well
        #     pf.add_parameters(filenames=list_file, par_type="grid", par_name_base="wel_grid_{0}".format(kper),
        #                       pargp="wel_{0}".format(kper), index_cols=[0, 1, 2], use_cols=[3],
        #                       upper_bound=1.5, lower_bound=0.5)
        #
        # # add grid-scale parameters for SFR reach conductance.  Use layer, row, col and reach
        # # number in the parameter names
        # pf.add_parameters(filenames="freyberg6.sfr_packagedata.txt", par_name_base="sfr_rhk",
        #                   pargp="sfr_rhk", index_cols=[0,1,2,3], use_cols=[9], upper_bound=10.,
        #                   lower_bound=0.1,
        #                   par_type="grid")

        # add observations from the sfr observation output file
        df = pd.read_csv(os.path.join(org_ws, "sfr.csv"), index_col=0)
        pf.add_observations("sfr.csv", insfile="sfr.csv.ins", index_cols="time",
                            use_cols=list(df.columns.values),
                            prefix="sfr")

        # add observations for the heads observation output file
        df = pd.read_csv(os.path.join(org_ws, "heads.csv"), index_col=0)
        pf.add_observations("heads.csv", insfile="heads.csv.ins",
                            index_cols="time", use_cols=list(df.columns.values),
                            prefix="hds")
        # add model run command
        pf.mod_sys_cmds.append("mf6")
        shutil.copy(os.path.join(exe_dir, mf_exe),
                    os.path.join(pf.new_d, mf_exe))

    else:
        pf.mod_py_cmds.append("print('model')")
    pf.add_py_function("workflow.py",
                       call_str="log_array_files()",
                       is_pre_cmd=False)
    # build pest control file
    pst = pf.build_pst('freyberg.pst')
    shutil.copy2(os.path.join(exe_dir, pst_exe),
                 os.path.join(template_ws, pst_exe))

    # draw from the prior and save the ensemble in binary format
    pe = pf.draw(num_reals, use_specsim=True)
    pe.to_binary(os.path.join(template_ws, "prior.jcb"))

    # set some algorithmic controls
    pst.control_data.noptmax = 0
    pst.pestpp_options["additional_ins_delimiters"] = ","

    # write the control file
    pst.write(os.path.join(pf.new_d, "freyberg.pst"),version=2)

    # run with noptmax = 0
    pyemu.os_utils.run("{0} freyberg.pst".format(
        os.path.join("pestpp-ies")), cwd=pf.new_d)

    # make sure it ran
    res_file = os.path.join(pf.new_d, "freyberg.base.rei")
    assert os.path.exists(res_file), res_file
    pst.set_res(res_file)
    print(pst.phi)

    # if successful, set noptmax = -1 for prior-based Monte Carlo
    pst.control_data.noptmax = -1
    
    # define what file has the prior parameter ensemble
    pst.pestpp_options["ies_par_en"] = "prior.jcb"
    pst.pestpp_options['save_binary'] = True
    pst.pestpp_options["ies_num_reals"] = num_reals

    # write the updated pest control file
    pst.write(os.path.join(pf.new_d, "freyberg.pst"),version=2)
    shutil.copy(os.path.join(pf.new_d, "freyberg.obs_data.csv"),
                os.path.join(pf.new_d, "freyberg.obs_data_orig.csv"))
    #assert pst.phi < 1.0e-5, pst.phi


def run(t_d,num_workers=5,num_reals=100,noptmax=-1,m_d=None,init_lam=None):
    if m_d is None:
        m_d = t_d.replace("template","master")
    pst = pyemu.Pst(os.path.join(t_d,"freyberg.pst"))
    pst.pestpp_options["ies_num_reals"] = num_reals
    pst.control_data.noptmax = noptmax
    if init_lam is not None:
        pst.pestpp_options["ies_initial_lambda"] = init_lam
    #pst.pestpp_options["ies_bad_phi_sigma"] = 1.5
    pst.pestpp_options["ies_subset_size"] = -10
    pst.write(os.path.join(t_d,"freyberg.pst"),version=2)
    pyemu.os_utils.start_workers(t_d,"pestpp-ies","freyberg.pst",
                                 num_workers=num_workers,worker_root=".",
                                 master_dir=m_d,
                                 port=4199)
    return m_d

def make_kickass_figs(m_d_c = "master_flow_prior",m_d_f = "master_flow_post",
                      plt_name="histo_compare_pub.pdf"):

    unit_dict = {"head": "sw-gw flux $\\frac{ft^3}{d}$",
                 "tail": "sw-gw flux $\\frac{ft^3}{d}$",
                 "trgw": "gw level $ft$",
                 "gage": "sw flux $\\frac{ft^3}{d}$"}
    label_dict = {"head": "headwater",
                  "tail": "tailwater",
                  "trgw_2_29_5": "gw_1",
                  "trgw_2_101_23": "gw_2",
                  "gage": "sw_1"}

    #sim = flopy.mf6.MFSimulation.load(sim_ws=m_d_f)
    #m = sim.get_model("freyberg6")
    #redis_fac = m.dis.nrow.data / 40
    #redis_fac = 3

    pst = pyemu.Pst(os.path.join(m_d_c,"freyberg.pst"))
    obs = pst.observation_data
    obs = obs.loc[obs.otype=="lst",:]
    obs.loc[:,"time"] = obs.time.astype(float)
    grps = obs.obgnme.unique()
    grps.sort()

    hw_fore = obs.loc[obs.apply(lambda x: x.time==700. and "headwater" in x.obsnme,axis=1),"obsnme"]
    assert hw_fore.shape[0] == 1
    lay1_fore = obs.loc[obs.apply(lambda x: x.time==700. and "trgw_0_80_20" in x.obsnme,axis=1),"obsnme"]
    assert lay1_fore.shape[0] == 1
    lay3_fore = obs.loc[obs.apply(lambda x: x.time == 700. and "trgw_2_80_20" in x.obsnme, axis=1), "obsnme"]
    assert lay3_fore.shape[0] == 1

    oe_pr = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d_c,"freyberg.0.obs.jcb"))
    oe_pt= pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d_f,"freyberg.0.obs.jcb"))

    fig,axes = plt.subplots(1,3,figsize=(8,3))
    titles = ["A) layer 1 groundwater level","B) layer 3 groundwater level","C) headwater exchange flux"]
    labels = ["meters","meters","$\\frac{meters}{day}$"]
    for ax,fore,title,label in zip(axes,[lay1_fore,lay3_fore,hw_fore],titles, labels):
        ax.hist(oe_pr.loc[:,fore],facecolor="0.5",edgecolor="none",alpha=0.5)
        ax.hist(oe_pt.loc[:,fore], facecolor="b", edgecolor="none", alpha=0.5)
        ax.set_yticks([])
        ax.set_ylabel("probability density")
        ax.set_xlabel(label)
        ax.set_title(title,loc="left")
    plt.tight_layout()
    plt.savefig(plt_name+".pdf")
    plt.close(fig)

    def namer(name):
        if "trgw" in name:
            full_name = "groundwater level"
            k = int(name.split("_")[3])
            i = int(name.split("_")[4])
            j = int(name.split("_")[5])
            full_name += ", layer:{0}, row:{1}, column:{2}".format(k+1,i+1,j+1)
        elif "headwater" in name:
            full_name = "headwater surface-water/groundwater exchange flux"
        elif "tailwater" in name:
            full_name = "tailwater surface-water/groundwater exchange flux"
        elif "gage" in name:
            full_name = "surface-water flow"
        return full_name


    with PdfPages(plt_name+"_si.pdf") as pdf:
        ax_count = 0
        for grp in grps:
            gobs = obs.loc[obs.obgnme==grp,:].copy()
            gobs.sort_values(by="time",inplace=True)
            fig,ax = plt.subplots(1,1,figsize=(8,4))
            gtime = gobs.time.values.copy()
            gnames = gobs.obsnme.values.copy()
            [ax.plot(gtime,oe_pr.loc[i,gnames],"0.5",lw=0.1,alpha=0.5) for i in oe_pr.index]
            [ax.plot(gtime, oe_pt.loc[i, gnames], "b", lw=0.1,alpha=0.5) for i in oe_pt.index]
            ax.set_xlabel("time")
            ax.set_title("{0}) {1}".format(ax_count,namer(grp)),loc="left")
            ax_count += 1
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)



def write_par_sum(pst_file):
    pst = pyemu.Pst(pst_file)
    par = pst.parameter_data
    par.loc[par.pargp.str.startswith("wel"),"pargp"] = "wel"
    def group_namer(grp):
        name = ''
        if "layer" in grp:
            name += "Layer " + grp.split('layer')[1][0]
        if "gr" in grp:
            name += " grid-scale"
        elif "pp" in grp:
            name += " pilot points"
        elif "const" in grp:
            name += " constant"
        if "_k_" in grp:
            name += " HK"
        elif "k33" in grp:
            name += " VK"
        elif "ss" in grp:
            name += " SS"
        elif "sy" in grp:
            name += " SY"
        elif "rch" in grp:
            name += " recharge"
        if "sfr" in grp:
            name = " SFR stream-bed conductance"
        if "twel" in grp:
            name = "temporal wel flux constants"
        #elif "wel" in grp:
        #    name = "grid-scale wel flux for stress period " + grp.split('_')[1]
        elif "wel" in grp:
            name = "grid-scale wel flux"
        return name

    ugrps = pst.parameter_data.pargp.unique()
    name_dict = {ug:group_namer(ug) for ug in ugrps}
    pst.write_par_summary_table(os.path.split(pst_file)[0] + "_par_sum.tex",group_names=name_dict)


def set_obsvals_weights(t_d,double_ineq_ss=True):
    lines = open(os.path.join(t_d,"freyberg6.obs"),'r').readlines()
    ijs = []
    for line in lines:
        if line.lower().strip().startswith("trgw"):
            raw = line.split()
            i,j = int(raw[-2])-1,int(raw[-1])-1
            ijs.append((i,j))
    ijs = list(set(ijs))
    pst = pyemu.Pst(os.path.join(t_d, "freyberg.pst"))
    obs = pd.read_csv(os.path.join(t_d, "freyberg.obs_data_orig.csv"))
    obs = obs.set_index('obsnme', drop=False)
    obs.loc[:, "weight"] = 0.0
    obs.loc[:, "lower_bound"] = -30
    obs.loc[:, "upper_bound"] = 30
    pst.observation_data = obs
    obs = obs.loc[obs.otype == "arr", :].copy()
    obs.loc[:, "i"] = obs.i.astype(int)
    obs.loc[:, "j"] = obs.j.astype(int)
    obs.loc[:, "k"] = obs.obsnme.apply(lambda x: int(x.split("layer")[1][0])) - 1
    obs.loc[:, "ij"] = obs.apply(lambda x: (x.i, x.j), axis=1)
    obs.loc[:, "kij"] = obs.apply(lambda x: (x.k, x.i, x.j), axis=1)

    wdf = pd.read_csv(os.path.join(t_d, "freyberg6.wel_stress_period_data_1.txt"), header=None,
                      names=["l", "r", "c", "flux"], delim_whitespace=True)
    wdf.loc[:, "kij"] = wdf.apply(lambda x: (int(x.l) - 1, int(x.r) - 1, int(x.c) - 1), axis=1)
    wkij = set(wdf.kij.tolist())
    w_nznames = obs.loc[obs.apply(lambda x: x.kij in wkij and "npfk" in x.oname and "33" not in x.oname, axis=1), "obsnme"]
    for kij in wkij:
        exclude_i = np.arange(kij[1]-10,kij[1]+10,dtype=int)
        exclude_j = np.arange(kij[2] - 10, kij[2] + 10, dtype=int)
        exclude = []
        for i in exclude_i:
            exclude.extend([(i,j) for j in exclude_j])
        exclude = set(exclude)
        ijs = [ij for ij in ijs if ij not in exclude]
    np.random.seed(222)
    idxs = np.random.randint(0,len(ijs),4)
    ijs = [ijs[i] for i in idxs]
    #pst = pyemu.Pst(os.path.join(t_d,"freyberg.pst"))
    #obs = pd.read_csv(os.path.join(t_d, "freyberg.obs_data_orig.csv"))

    hk_nznames = obs.loc[obs.apply(
        lambda x:  x.ij in ijs and x.k != 1 and "npfk" in x.oname and "dup" not in x.oname and "33" not in x.oname, axis=1),
                         "obsnme"]
    ss_nznames = obs.loc[obs.apply(
        lambda x: x.ij in ijs and x.k != 1 and "stoss" in x.oname and "dup" not in x.oname, axis=1),
                         "obsnme"]
    dup_ss_nznames = ss_nznames.apply(lambda x: x.replace("sto","dup-sto"))
    missing = []
    names = set(obs.obsnme.tolist())
    for dname in dup_ss_nznames:
        if dname not in names:
            missing.append(dname)
    if len(missing) > 0:
        print(missing)
        raise Exception("missing dups...")

    # wdf = pd.read_csv(os.path.join(t_d,"freyberg6.wel_stress_period_data_1.txt"),header=None,names=["l","r","c","flux"],delim_whitespace=True)
    # wdf.loc[:,"kij"] = wdf.apply(lambda x: (int(x.l)-1,int(x.r)-1,int(x.c)-1),axis=1)
    # wkij = set(wdf.kij.tolist())
    # w_nznames = obs.loc[obs.apply(lambda x: x.kij in wkij and "npfk" in x.oname and "dup" not in x.oname,axis=1),"obsnme"]

    vals = np.random.normal(0,1.0,len(hk_nznames))
    #set one really low
    vals[-1] = -2.0
    vals = pst.observation_data.loc[hk_nznames,"obsval"].values + vals
    pst.observation_data.loc[hk_nznames, "obsval"] = vals

    pst.observation_data.loc[hk_nznames, "lower_bound"] = vals - 2
    pst.observation_data.loc[hk_nznames, "upper_bound"] = vals + 2
    pst.observation_data.loc[hk_nznames, "weight"] = 1.0 + np.cumsum(np.ones(len(hk_nznames))+1)

    vals = np.random.normal(1.5, 0.1, len(w_nznames))
    #vals = pst.observation_data.loc[w_nznames, "obsval"].values + vals
    pst.observation_data.loc[w_nznames, "obsval"] = vals
    pst.observation_data.loc[w_nznames, "lower_bound"] = vals - 1
    pst.observation_data.loc[w_nznames, "upper_bound"] = vals + 1
    pst.observation_data.loc[w_nznames, "weight"] = 10
    pst.observation_data.loc[w_nznames, "obgnme"] = obs.loc[w_nznames,"oname"].apply(lambda x: "greater_than_well_"+x)

    pst.observation_data.loc[:,"observed_value"] = pst.observation_data.obsval.values
    vals = np.random.normal(0, 1.0, len(ss_nznames))
    if double_ineq_ss is True:
        vals = pst.observation_data.loc[ss_nznames, "obsval"].values + vals
        pst.observation_data.loc[ss_nznames, "obsval"] = vals - 0.75
        pst.observation_data.loc[dup_ss_nznames, "obsval"] = vals + 0.75
        pst.observation_data.loc[ss_nznames, "obgnme"] = obs.loc[ss_nznames,"oname"].apply(lambda x: "greater_than_"+x)
        pst.observation_data.loc[dup_ss_nznames, "obgnme"] = obs.loc[dup_ss_nznames,"oname"].apply(lambda x: "less_than_"+x)
        pst.observation_data.loc[ss_nznames, "weight"] = 10.0
        pst.observation_data.loc[dup_ss_nznames, "weight"] = 10.0
        pst.observation_data.loc[ss_nznames,"observed_value"] = vals
        pst.observation_data.loc[dup_ss_nznames, "observed_value"] = vals

    else:

        vals = pst.observation_data.loc[ss_nznames, "obsval"].values + vals
        pst.observation_data.loc[ss_nznames, "obsval"] = vals
        pst.observation_data.loc[ss_nznames, "lower_bound"] = vals - 1.5
        pst.observation_data.loc[ss_nznames, "upper_bound"] = vals + 1.5
        pst.observation_data.loc[ss_nznames, "weight"] =  1.0 + np.cumsum(np.ones(len(ss_nznames))+1)
        pst.observation_data.loc[ss_nznames, "observed_value"] = vals

    nzobs = pst.observation_data.loc[pst.nnz_obs_names]
    mean_vals = pst.observation_data.loc[:,"obsval"]

    cov = pyemu.Cov.from_observation_data(pst)
    df = pyemu.Ensemble._gaussian_draw(
        cov=cov,
        mean_values=mean_vals,
        num_reals=pst.pestpp_options['ies_num_reals'],
        grouper=None,
        fill=False,
        factor="eigen",
    )
    obs = pst.observation_data
    # lb_dict = obs.lower_bound.to_dict()
    # ub_dict = obs.upper_bound.to_dict()
    # for oname in nzobs.obsnme:
    #     vals = df.loc[:,oname].values
    #     vals[vals<lb_dict[oname]] = lb_dict[oname]
    #     vals[vals > ub_dict[oname]] = ub_dict[oname]
    #     df.loc[:,oname] = vals
    pyemu.ObservationEnsemble(pst, df).to_binary(
        os.path.join(t_d, "freyberg.obs+noise_0.jcb")
    )
    pst.pestpp_options['ies_observation_ensemble'] = "freyberg.obs+noise_0.jcb"

    #noise = np.random.normal(0,2,(1000,len(hk_nznames)))

    pst.write(os.path.join(t_d,"freyberg.pst"),version=2)
    print(pst.observation_data.loc[
              pst.nnz_obs_names,
              ["obsval", "weight", "lower_bound", "upper_bound"]])

    with PdfPages("noise.pdf") as pdf:
        for oname in nzobs.obsnme:
            fig,ax = plt.subplots(1,1)
            df.loc[:,oname].hist(ax=ax,facecolor="r",alpha=0.5)
            ylim = ax.get_ylim()
            ax.plot([nzobs.loc[oname,"obsval"],nzobs.loc[oname,"obsval"]],ylim,"r")
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)



def plot_fields(m_d):
    ib = np.loadtxt(os.path.join(m_d,"freyberg6.dis_idomain_layer1.txt"),dtype=int)
    pst = pyemu.Pst(os.path.join(m_d,"freyberg.pst"))
    obs = pst.observation_data.copy()
    obs = obs.loc[obs.otype=="arr",:]
    obs.loc[:,"i"] = obs.i.astype(int)
    obs.loc[:, "j"] = obs.j.astype(int)
    obs.loc[:,"k"] = obs.obsnme.apply(lambda x: int(x.split("layer")[1][0])-1)
    onames = obs.oname.unique()
    onames.sort()
    pr_oe = pd.read_csv(os.path.join(m_d,"freyberg.0.obs.csv"),index_col=0,nrows=100)
    pt_oe = pd.read_csv(os.path.join(m_d, "freyberg.4.obs.csv".\
                                     format(pst.control_data.noptmax)), index_col=0,nrows=100)

    with PdfPages(os.path.join(m_d,"results_fields.pdf")) as pdf:
        for oname in onames:
            oobs = obs.loc[obs.oname==oname,:].copy()
            for real in pt_oe.index.values[:10]:
                pra = np.zeros(ib.shape)
                pta = np.zeros(ib.shape)
                pra[oobs.i,oobs.j] = pr_oe.loc[real,oobs.obsnme]
                pta[oobs.i, oobs.j] = pt_oe.loc[real, oobs.obsnme]
                pra[ib<=0] = np.nan
                pta[ib <= 0] = np.nan
                mn =min(np.nanmin(pra),np.nanmin(pta))
                mx = max(np.nanmax(pra), np.nanmax(pta))

                fig,axes = plt.subplots(1,2,figsize=(8,4))
                axes[0].imshow(pra,vmin=mn,vmax=mx)
                axes[1].imshow(pta, vmin=mn, vmax=mx)
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)
                break
            break


def log_array_files(d='.'):
    mult_df = pd.read_csv(os.path.join(d, "mult2model_info.csv"), index_col=0)
    for f in mult_df.model_file.unique():
        ar = np.loadtxt(os.path.join(d, f))
        np.savetxt(os.path.join(d, f"log_{f}"), np.log10(ar))
        np.savetxt(os.path.join(d, f"log_dup_{f}"), np.log10(ar))


def build_localizer(t_d):
    pst = pyemu.Pst(os.path.join(t_d,"freyberg.pst"))
    par = pst.parameter_data.loc[pst.adj_par_names,:].copy()
    obs = pst.observation_data.loc[pst.nnz_obs_names,:].copy()
    print(par.pname.unique())
    print(obs.oname.unique())
    onames = set(obs.oname.unique())
    par = par.loc[par.pname.apply(lambda x: x in onames)]
    ogp = obs.obgnme.unique()
    ogp.sort()
    pgp = par.pargp.unique()
    pgp.sort()
    df = pd.DataFrame(index=ogp,columns=pgp,dtype=float)
    df.loc[:,:] = 0.0
    obs.loc[:,"oname_dedup"] = obs.oname.apply(lambda x: x.replace("dup-",""))
    for name in onames:
        ppar = par.loc[par.pname==name,:].copy()
        oobs = obs.loc[obs.oname_dedup==name,:].copy()
        ppgp = ppar.pargp.unique()
        ppgp.sort()
        oogp = oobs.obgnme.unique()
        oogp.sort()
        print(name,ppgp,oogp)
        df.loc[oogp,ppgp] = 1.0
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    x = df.values.copy()
    x[x==0.0] = np.nan
    ax.imshow(x)
    ax.set_xticks(np.arange(len(pgp)))
    ax.set_yticks(np.arange(len(ogp)))
    ax.set_xticklabels(pgp,rotation=90)
    ax.set_yticklabels(ogp)
    plt.tight_layout()
    plt.savefig(os.path.join(t_d,"loc.pdf"))
    plt.close(fig)

    df.to_csv(os.path.join(t_d,"loc.csv"))
    pst.pestpp_options["ies_localizer"] = "loc.csv"
    pst.control_data.noptmax = -2
    pst.write(os.path.join(t_d,"freyberg.pst"),version=2)
    pyemu.os_utils.run("pestpp-ies freyberg.pst",cwd=t_d)

def transfer_pars(cond_pst_file,cond_pe_file,flow_t_d,joint_pe_file):
    pst = pyemu.Pst(os.path.join(flow_t_d,"freyberg.pst"))
    flow_pe_file = pst.pestpp_options["ies_par_en"]
    flow_pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(flow_t_d,flow_pe_file))
    cond_pst = pyemu.Pst(cond_pst_file)
    cond_pe = pyemu.ParameterEnsemble.from_binary(pst=cond_pst,filename=cond_pe_file)
    cond_names = set(cond_pe.index.to_list())
    flow_names = set(flow_pe.index.to_list())
    common = cond_names.intersection(flow_names)
    common = list(common)
    common.sort()
    cond_pe = cond_pe.loc[common,:]
    flow_pe = flow_pe.loc[common,:]

    #if "base" in cond_pe.index:
    #    cond_pe = cond_pe.loc[cond_pe.index.map(lambda x: "base" not in x),:]

    cond_names = set(cond_pe.columns.to_list())
    flow_names = set(flow_pe.columns.to_list())
    d = cond_names - flow_names
    assert len(d) == 0,d
    flow_pe = flow_pe.loc[cond_pe.index,:]
    flow_pe.loc[:,cond_pe.columns] = cond_pe.loc[:,:]
    print(cond_pe.shape,flow_pe.shape)

    flow_pe.to_binary(os.path.join(flow_t_d,joint_pe_file))
    pst.pestpp_options["ies_par_en"] = joint_pe_file
    pst.control_data.noptmax = -2
    pst.write(os.path.join(flow_t_d,"freyberg.pst"),version=2)
    pyemu.os_utils.run("pestpp-ies freyberg.pst",cwd=flow_t_d)


def plot_mult(t_d):

    df = pd.read_csv(os.path.join(t_d, "mult2model_info.csv"))
    df = df.loc[df.model_file.str.contains("npf_k_layer1"), :]
    #print(df)
    #return
    pst = pyemu.Pst(os.path.join(t_d,"freyberg.pst"))
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(t_d,"prior.jcb")).loc[:,pst.par_names]
    pst.parameter_data.loc[:,"parval1"] = pe.iloc[0,:].values
    par = pst.parameter_data
    pps = par.loc[(par.ptype=='pp') & par.parnme.str.contains("npfklayer1")]
    pst.write_input_files(pst_path=t_d)
    os.chdir(t_d)
    pyemu.helpers.apply_list_and_array_pars()
    os.chdir("..")
    fig,axes = plt.subplots(1,df.shape[0]+2,figsize=(8.5,2))
    org_arr = np.log10(np.loadtxt(os.path.join(t_d,df.org_file.iloc[0])))
    ib = np.loadtxt(os.path.join(t_d, "freyberg6.dis_idomain_layer1.txt")).reshape(org_arr.shape)
    org_arr[ib<=0] = np.nan
    new_arr = org_arr.copy()

    mx,mn = -1.0e+30,1.0e+30
    for ax,fname in zip(axes[1:-1],df.mlt_file):
        arr = np.log10(np.loadtxt(os.path.join(t_d,fname)))
        arr[ib <= 0] = np.nan
        mx = max(mx,np.nanmax(arr))
        mn = min(mn,np.nanmin(arr))

        new_arr *= arr

    cb = axes[0].imshow(org_arr,vmin=np.nanmin(new_arr),vmax=np.nanmax(new_arr))
    plt.colorbar(cb,ax=axes[0],label="$log_{10} \\frac{m}{d}$")
    axes[0].set_title("A) original model input",loc="left")
    labs = ["B) grid-scale","C) pilot points","D) layer constant"]
    for ax, fname,lab in zip(axes[1:-1], df.mlt_file,labs):
        arr = np.log10(np.loadtxt(os.path.join(t_d, fname)))
        arr[ib <= 0] = np.nan
        cb = ax.imshow(arr,vmin=mn,vmax=mx, interpolation='none')
        plt.colorbar(cb, ax=ax, label="$log_{10}$ multiplier")
        if 'pilot' in lab:
            ax.scatter(pps.j.astype(int), pps.i.astype(int), fc='none', marker='o',
                       ec='r', s=3, lw=0.2)
        ax.set_title(lab,loc="left")


    cb = axes[-1].imshow(new_arr)
    plt.colorbar(cb, ax=axes[-1], label="$log_{10} \\frac{m}{d}$")
    axes[-1].set_title("E) new model input")
    for ax in axes:
        ax.set_xlabel("column")
        ax.set_ylabel("row")
    plt.tight_layout()
    plt.savefig("mult.pdf")
    plt.close(fig)



def plot_domain():
    ib = np.loadtxt(os.path.join("daily_template", "freyberg6.dis_idomain_layer1.txt")).reshape((120,60))
    ib[ib>0] = np.nan
    pst_c = pyemu.Pst(os.path.join("daily_template_cond","freyberg.pst"))
    pst = pyemu.Pst(os.path.join("daily_template","freyberg.pst"))
    obs_c = pst_c.observation_data.loc[pst_c.nnz_obs_names,:].copy()
    #print(obs_c.obgnme.unique())
    obs_c.loc[:,'k'] = obs_c.oname.apply(lambda x: int(x[-1])-1)
    obs_c .loc[:,"ij"] = obs_c.apply(lambda x: (int(x.i),int(x.j)),axis=1)
    c_grp = obs_c.groupby("k").ij.unique().to_dict()
    g_grp = obs_c.groupby("ij").obgnme.unique().to_dict()
    #print(g_grp)

    obs = pst.observation_data
    obs = obs.loc[obs.usecol=="trgw",:]
    obs.loc[:,"k"] = obs.obsnme.apply(lambda x: int(x.split("_")[3]))
    obs.loc[:, "i"] = obs.obsnme.apply(lambda x: int(x.split("_")[4]))
    obs.loc[:, "j"] = obs.obsnme.apply(lambda x: int(x.split("_")[5]))
    obs.loc[:,"ij"] = obs.apply(lambda x: [x.i,x.j],axis=1)
    #print(obs.obgnme.unique())

    sfr_arr = np.zeros_like(ib,dtype=float)
    sfr_arr[:60, 47] = 1
    sfr_arr[60:, 47] = 2

    #sfr_arr = np.ma.masked_where(sfr_arr == 0,sfr_arr)
    #print(np.unique(sfr_arr))
    #plt.imshow(sfr_arr)
    # axes[0].scatter([47],[118],marker="^",color="k",s=30)
    #axes[0].set_ylim(119, 0)

    fig,axes = plt.subplots(1,5,figsize=(8,3))
    for i,ax in enumerate(axes[:-1]):

        ax.imshow(ib,cmap="Greys_r",alpha=0.5)

        ax.scatter(np.nan, np.nan, c='0.35', marker='s', label='inactive domain')

        ax.scatter(np.nan, np.nan, c='b', marker='|', label="headwater reaches")
        ax.scatter(np.nan, np.nan, c='g', marker='|', label="tailwater reaches")

        ax.scatter([20], [80], marker="^", c='b', s=30, label="GW level forecast")
        # ax.scatter([20], [80], marker="^", c='b', s=30, label="GW level forecast")
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        if i % 2 == 0:
            ax.plot([49,49],[0,60],"b")
            ax.plot([49, 49],[60,119], "g")

        #ax.imshow(sfr_arr, cmap="winter")

    tags = ["npfklayer1","npfklayer3","stosslayer1","stosslayer3"]
    names = ["A) Layer 1 HK","B) Layer 3 HK","C) Layer 1 SS","D) Layer 3 SS"]
    for tag,ax,name in zip(tags,axes,names):
        k = int(tag[-1]) - 1
        tobs = obs_c.loc[obs_c.obgnme.str.contains(tag),:].copy()
        ijs = tobs.loc[tobs.obgnme.apply(lambda x: "less" not in x and "greater" not in x),:]
        print(tag,ijs.obgnme)
        ax.scatter([ij[1] for ij in ijs.ij],[ij[0] for ij in ijs.ij],marker = ".",color='r',label="equality")
        ijs = tobs.loc[tobs.obgnme.str.contains("less"),:]
        ax.scatter([ij[1] for ij in ijs.ij], [ij[0] for ij in ijs.ij], marker="|", color='r', label="less-than inequality")
        ijs = tobs.loc[tobs.obgnme.str.contains("greater"), :]
        ax.scatter([ij[1] for ij in ijs.ij], [ij[0] for ij in ijs.ij], marker="_", color='r',
                   label="greater-than inequality")
        ax.set_title(name,loc="left")
    axes[-1].set_axis_off()


    plt.tight_layout()
    ax.legend(bbox_to_anchor=(1.1,1))
    plt.savefig("domain.pdf")
    return

    ivals = [ij[0] for ij in c_grp[0] if True in [True if "greater_than" in grp else False for grp in g_grp[ij]]]
    jvals = [ij[1] for ij in c_grp[0] if True in [True if "greater_than" in grp else False for grp in g_grp[ij]]]
    axes[0].scatter(jvals,ivals,marker=">",color="r",s=20,
                    label="greater-than inequality obs")
    #ivals = [ij[0] for ij in c_grp[0] if "less_than" in g_grp[ij]]
    #jvals = [ij[1] for ij in c_grp[0] if "less_than" in g_grp[ij]]
    ivals = [ij[0] for ij in c_grp[0] if True in [True if "less_than" in grp else False for grp in g_grp[ij]]]
    jvals = [ij[1] for ij in c_grp[0] if True in [True if "less_than" in grp else False for grp in g_grp[ij]]]
    axes[0].scatter(jvals, ivals, marker="P", color="r", s=100,

                    label="less-than inequality obs")
    #ivals = [ij[0] for ij in c_grp[0] if "greater_than" not in g_grp[ij] and "less_than" not in g_grp[ij]]
    #jvals = [ij[1] for ij in c_grp[0] if "greater_than" not in g_grp[ij] and "less_than" not in g_grp[ij]]
    ivals = [ij[0] for ij in c_grp[0] if True not in [True if "greater_than" in grp and "less_than" in grp else False for grp in g_grp[ij]]]
    jvals = [ij[1] for ij in c_grp[0] if True not in [True if "greater_than" in grp and "less_than" in grp else False for grp in g_grp[ij]]]
    axes[0].scatter(jvals, ivals, marker="o", color="r", s=20,
                    label="equality obs")

    axes[1].imshow(ib,cmap="Greys_r",alpha=0.25)
    #ivals = [ij[0] for ij in c_grp[2] if "greater_than" in g_grp[ij]]
    #jvals = [ij[1] for ij in c_grp[2] if "greater_than" in g_grp[ij]]
    ivals = [ij[0] for ij in c_grp[2] if True in [True if "greater_than" in grp else False for grp in g_grp[ij]]]
    jvals = [ij[1] for ij in c_grp[2] if True in [True if "greater_than" in grp else False for grp in g_grp[ij]]]
    axes[1].scatter(jvals, ivals, marker=">", color="r", s=20,
                    label="greater-than inequality obs")
    #ivals = [ij[0] for ij in c_grp[0] if "less_than" in g_grp[ij]]
    #jvals = [ij[1] for ij in c_grp[0] if "less_than" in g_grp[ij]]
    ivals = [ij[0] for ij in c_grp[2] if True in [True if "less_than" in grp else False for grp in g_grp[ij]]]
    jvals = [ij[1] for ij in c_grp[2] if True in [True if "less_than" in grp else False for grp in g_grp[ij]]]
    axes[1].scatter(jvals, ivals, marker="P", color="r", s=100,
                    label="less-than inequality obs")
    #ivals = [ij[0] for ij in c_grp[2] if "greater_than" not in g_grp[ij]]
    #jvals = [ij[1] for ij in c_grp[2] if "greater_than" not in g_grp[ij]]
    ivals = [ij[0] for ij in c_grp[2] if
             True not in [True if "greater_than" in grp and "less_than" in grp else False for grp in g_grp[ij]]]
    jvals = [ij[1] for ij in c_grp[2] if
             True not in [True if "greater_than" in grp and "less_than" in grp else False for grp in g_grp[ij]]]
    axes[1].scatter(jvals, ivals, marker="o", color="r", s=20,
                    label="equality obs")

    sfr_arr = np.zeros_like(ib)
    sfr_arr[:60,47] = 1
    sfr_arr[60:, 47] = 2

    sfr_arr[sfr_arr==0] = np.nan
    axes[0].imshow(sfr_arr,cmap="winter")
    axes[0].scatter(np.nan, np.nan, c='b', marker='|', label="headwater reaches")
    axes[0].scatter([20],[80],marker="^",c='b',s=30, label="level forecast")
    axes[1].scatter([20], [80], marker="^", c='b', s=30, label="level forecast")

    #axes[0].scatter([47],[118],marker="^",color="k",s=30)
    axes[0].set_ylim(119,0)
    #axes[0].legend()
    #axes[1].imshow(sfr_arr)
    for ax in axes:
        ax.set_ylabel("row")
        ax.set_xlabel("column")

    axes[0].set_title("A) layer 1",loc="left")
    axes[1].set_title("B) layer 3",loc="left")

    plt.tight_layout()
    plt.savefig("domain.pdf")
    plt.close(fig)



def mod_to_stationary(org_t_d):
    new_t_d = org_t_d+"_stat"
    if os.path.exists(new_t_d):
        shutil.rmtree(new_t_d)
    shutil.copytree(org_t_d,new_t_d)
    pst = pyemu.Pst(os.path.join(new_t_d,"freyberg.pst"))
    obs = pst.observation_data
    obs.loc[obs.obgnme.str.startswith("less") & obs.oname.str.contains("npf"),"weight"] = 0.0
    obs.loc[obs.obgnme.str.startswith("greater") & obs.oname.str.contains("npf"), "weight"] = 0.0
    obs.loc[obs.obsnme.str.contains("dup"), "weight"] = 0.0

    obs.loc[obs.obgnme.str.contains("less") | obs.obgnme.str.contains("greater"), "obgnme"] = obs.loc[obs.obgnme.str.contains("less") | obs.obgnme.str.contains("greater"), "oname"].values

    obs = pst.observation_data.loc[pst.nnz_obs_names,:]
    onames = obs.oname.unique()
    for oname in onames:
        obs.loc[obs.oname==oname,"weight"] = obs.loc[obs.oname==oname,"weight"].max()

    print(obs.loc[pst.nnz_obs_names,"obgnme"].unique())
    pst.write(os.path.join(new_t_d,"freyberg.pst"),version=2)
    return new_t_d


if __name__ == "__main__":

    #setup_interface("temp_daily_test",num_reals=500,full_interface=False,include_constants=True)
    #set_obsvals_weights("daily_template_cond")
    #m_d = run("daily_template_cond", num_workers=50, num_reals=500, noptmax=6, init_lam=-0.1)
    #build_localizer(new_t_d)
    #m_d = run(new_t_d,num_workers=50,num_reals=500,noptmax=6,init_lam=-0.1)
    processing.plot_results_pub("daily_master_cond", pstf="freyberg", log_oe=False,noptmax=6)
    #processing.plot_histo_pub("daily_master_cond", pstf="freyberg", log_oe=False, noptmax=6)
    # cond_m_d = "daily_master_cond"
    # transfer_pars(os.path.join(cond_m_d,"freyberg.pst"),
    #               os.path.join(cond_m_d,"freyberg.0.par.jcb"),
    #               "daily_template","cond_prior.jcb")
    # run("daily_template",num_workers=15,num_reals=100,noptmax=-1,m_d="master_flow_prior")
    # transfer_pars(os.path.join(cond_m_d,"freyberg.pst"),
    #               os.path.join(cond_m_d,"freyberg.6.par.jcb"),
    #               "daily_template","cond_post.jcb")
    # make_kickass_figs()
    #processing.plot_histo("daily_master_cond", pstf="freyberg", log_oe=False, noptmax=6)
    # processing.plot_par_changes("daily_master_cond")

    # new_t_d = mod_to_stationary("daily_template_cond")
    # build_localizer(new_t_d)

    # cond_m_d = "daily_master_cond_stat"
    # transfer_pars(os.path.join(cond_m_d,"freyberg.pst"),
    #                os.path.join(cond_m_d,"freyberg.0.par.jcb"),
    #                "daily_template","cond_prior.jcb")
    # run("daily_template",num_workers=15,num_reals=100,noptmax=-1,m_d="master_flow_prior_stat")
    # transfer_pars(os.path.join(cond_m_d,"freyberg.pst"),
    #                os.path.join(cond_m_d,"freyberg.6.par.jcb"),
    #                "daily_template","cond_post.jcb")
    # run("daily_template", num_workers=15, num_reals=100, noptmax=-1, m_d="master_flow_post_stat")
    #make_kickass_figs(m_d_c="master_flow_prior_stat",
    #                  m_d_f="master_flow_post_stat",plt_name="flow_stat")
    # plot_mult("daily_template_cond")
    #plot_domain()

