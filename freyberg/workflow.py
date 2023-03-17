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
    exe_dir = os.path.join("bin_new","win")
    mf_exe = 'mf6.exe'
    pst_exe = 'pestpp-ies.exe'

elif sys.platform.startswith('linux'):
    exe_dir = os.path.join("bin_new","linux")

    mf_exe = 'mf6'
    pst_exe = 'pestpp-ies'

    os.system(f'chmod +x {os.path.join(exe_dir, mf_exe)}')
    os.system(f'chmod +x {os.path.join(exe_dir, pst_exe)}')

elif sys.platform.lower().startswith('dar') or sys.platform.lower().startswith('mac'):
    exe_dir = os.path.join("bin_new","mac")
    mf_exe = 'mf6'
    pst_exe = 'pestpp-ies'

    os.system(f'chmod +x {os.path.join(exe_dir, mf_exe)}')
    os.system(f'chmod +x {os.path.join(exe_dir, pst_exe)}')
else:
    raise Exception('***ERROR: OPERATING SYSTEM UNKNOWN***')


def setup_interface(org_ws,num_reals=100,full_interface=True,include_constants=True,
    grid_gs=None,pp_gs=None, dir_suffix="template"):

    if os.path.exists(os.path.join(org_ws, mf_exe)):
        os.remove(os.path.join(org_ws, mf_exe))
    shutil.copy(os.path.join(exe_dir, mf_exe), os.path.join(org_ws, mf_exe))
    if os.path.exists(os.path.join(org_ws, pst_exe)):
        os.remove(os.path.join(org_ws, pst_exe))
    shutil.copy(os.path.join(exe_dir, pst_exe), os.path.join(org_ws, pst_exe))
    if os.path.exists(os.path.join(org_ws, "pyemu")):
        shutil.rmtree(os.path.join(org_ws, "pyemu"),ignore_errors=True)
    shutil.copytree("pyemu", os.path.join(org_ws, "pyemu"))
    if os.path.exists(os.path.join(org_ws, "flopy")):
        shutil.rmtree(os.path.join(org_ws, "flopy"),ignore_errors=True)
    shutil.copytree("flopy", os.path.join(org_ws, "flopy"))

    # load the mf6 model with flopy to get the spatial reference
    sim = flopy.mf6.MFSimulation.load(sim_ws=org_ws)
    m = sim.get_model("freyberg6")

    sim.simulation_data.max_columns_of_data = m.modelgrid.ncol

    # work out the spatial rediscretization factor
    redis_fac = m.dis.nrow.data / 40


    # where the pest interface will be constructed
    template_ws = org_ws.split('_')[1] + "_"+dir_suffix
    if not full_interface:
        template_ws += "_cond"
    
    
    # instantiate PstFrom object
    pf = pyemu.utils.PstFrom(original_d=org_ws, new_d=template_ws,
                remove_existing=True,
                longnames=True, spatial_reference=m.modelgrid,
                zero_based=False,start_datetime="1-1-2018")

    # the geostruct object for grid-scale parameters
    if grid_gs is None:
        grid_v = pyemu.geostats.ExpVario(contribution=1.0,a=500)
        grid_gs = pyemu.geostats.GeoStruct(variograms=grid_v)

    # the geostruct object for pilot-point-scale parameters
    if pp_gs is None:
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
    use_specsim=True
    if grid_gs.variograms[0].anisotropy != 1.0:
        use_specsim = False
    pe = pf.draw(num_reals, use_specsim=use_specsim)
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
    return pf.new_d


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
        ax.hist(oe_pr.loc[:,fore],facecolor="0.5",edgecolor="none",alpha=0.5,density=True)
        ax.hist(oe_pt.loc[:,fore], facecolor="b", edgecolor="none", alpha=0.5,density=True)
        ax.set_yticks([])
        ax.set_ylabel("probability density")
        ax.set_xlabel(label)
        ax.set_title(title,loc="left")
    plt.tight_layout()
    plt.savefig(plt_name)
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

    forecasts = ["trgw_0_80_20","trgw_2_80_20","headwater"]
    with PdfPages("flow_results_si.pdf") as pdf:
        ax_count = 0
        #for grp in grps:
        for fore,title,label in zip(forecasts,titles, labels):
            gobs = obs.loc[obs.obsnme.str.contains(fore),:].copy()
            gobs.sort_values(by="time",inplace=True)
            fig,ax = plt.subplots(1,1,figsize=(8,4))
            gtime = gobs.time.values.copy()
            gnames = gobs.obsnme.values.copy()
            [ax.plot(gtime,oe_pr.loc[i,gnames].values,"0.5",lw=0.1,alpha=0.5) for i in oe_pr.index]
            [ax.plot(gtime, oe_pt.loc[i, gnames].values, "b", lw=0.1,alpha=0.5) for i in oe_pt.index]
            ax.set_xlabel("time")
            ax.set_ylabel(label)
            ax.set_title("{0}".format(title),loc="left")
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
    #np.random.seed(222)
    np.random.seed(555)
    #idxs = np.random.randint(0,len(ijs),4)
    #print(ijs)
    #exit()
    #ijs = [ijs[i] for i in idxs]
    
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

    hk_iq_nznames = hk_nznames[::4]
    hk_nznames = [n for n in hk_nznames if n not in hk_iq_nznames]

    vals = np.random.normal(0,1.0,len(hk_nznames))
    #set one really low
    #vals[-1] = -2.0
    vals = pst.observation_data.loc[hk_nznames,"obsval"].values + vals
    vals[vals > 2.2] = 2.2
    vals[vals < 0.0] = 0.0
    
    pst.observation_data.loc[hk_nznames, "obsval"] = vals

    pst.observation_data.loc[hk_nznames, "lower_bound"] = vals - 2
    pst.observation_data.loc[hk_nznames, "upper_bound"] = vals + 2
    pst.observation_data.loc[hk_nznames, "weight"] = 1.0 + np.cumsum(np.ones(len(hk_nznames))+1)

    vals = pst.observation_data.loc[hk_iq_nznames,"obsval"] - 0.5
    pst.observation_data.loc[hk_iq_nznames, "obsval"] = vals
    #pst.observation_data.loc[hk_iq_nzname, "lower_bound"] = val - 1
    #pst.observation_data.loc[hk_iq_nzname, "upper_bound"] = val + 1
    pst.observation_data.loc[hk_iq_nznames, "weight"] = 20.0
    pst.observation_data.loc[hk_iq_nznames, "obgnme"] = obs.loc[hk_iq_nznames,"oname"].apply(lambda x: "less_than_"+x)
    
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
        skip_ijs = [obs.loc[n,"ij"] for n in hk_iq_nznames]
        ss_nznames = [n for n in ss_nznames if obs.loc[n,"ij"] not in skip_ijs]
        dup_ss_nznames = [n.replace("sto","dup-sto") for n in ss_nznames]
        vals = np.random.normal(0, 1.0, len(ss_nznames))
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
    #print(df.loc[:,hk_iq_nznames])
    
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
            df.loc[:,oname].hist(ax=ax,facecolor="r",alpha=0.5,density=True)
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


def plot_mult(t_d,plt_name="mult.pdf"):

    df = pd.read_csv(os.path.join(t_d, "mult2model_info.csv"))
    df = df.loc[df.model_file.str.contains("npf_k_layer1"), :]
    #print(df)
    #return
    pst = pyemu.Pst(os.path.join(t_d,"freyberg.pst"))
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(t_d,"prior.jcb")).loc[:,pst.par_names]
    pst.parameter_data.loc[:,"parval1"] = pe.iloc[1,:].values
    par = pst.parameter_data
    pps = par.loc[(par.ptype=='pp') & par.parnme.str.contains("npfklayer1")]
    pst.write_input_files(pst_path=t_d)
    os.chdir(t_d)
    pyemu.helpers.apply_list_and_array_pars()
    os.chdir("..")
    fig,axes = plt.subplots(1,df.shape[0]+2,figsize=(8.5,2))
    #org_arr = np.log10(np.loadtxt(os.path.join(t_d,df.org_file.iloc[0])))
    org_arr = np.loadtxt(os.path.join(t_d,df.org_file.iloc[0]))
    ib = np.loadtxt(os.path.join(t_d, "freyberg6.dis_idomain_layer1.txt")).reshape(org_arr.shape)
    org_arr[ib<=0] = np.nan
    new_arr = org_arr.copy()

    mx,mn = -1.0e+30,1.0e+30
    for ax,fname in zip(axes[1:-1],df.mlt_file):
        arr = np.log10(np.loadtxt(os.path.join(t_d,fname)))
        arr[ib <= 0] = np.nan
        mx = max(mx,np.nanmax(arr))
        mn = min(mn,np.nanmin(arr))

        new_arr *= 10**arr

    new_arr = np.log10(new_arr)
    org_arr = np.log10(org_arr)

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
    plt.savefig(plt_name)
    plt.close(fig)



def plot_domain(c_t_d="daily_template_cond",t_d="daily_template"):
    ib = np.loadtxt(os.path.join(t_d, "freyberg6.dis_idomain_layer1.txt")).reshape((120,60))
    ib[ib>0] = np.nan
    pst_c = pyemu.Pst(os.path.join(c_t_d,"freyberg.pst"))
    pst = pyemu.Pst(os.path.join(t_d,"freyberg.pst"))
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


def ensemble_stacking_experiment():

    grid_v = pyemu.geostats.ExpVario(contribution=1.0,a=2000,anisotropy=10,bearing=45)
    grid_gs = pyemu.geostats.GeoStruct(variograms=grid_v)    
    pp_v = pyemu.geostats.ExpVario(contribution=1.0, a=5000)
    pp_gs = pyemu.geostats.GeoStruct(variograms=pp_v)
    t_d = setup_interface("freyberg_daily",num_reals=100,full_interface=False,include_constants=True,grid_gs=grid_gs,pp_gs=pp_gs,dir_suffix="aniso1")
    plot_mult(t_d,plt_name="mult_aniso1.pdf")

    grid_v = pyemu.geostats.ExpVario(contribution=1.0,a=2000,anisotropy=10,bearing=45+90)
    grid_gs = pyemu.geostats.GeoStruct(variograms=grid_v)    
    pp_v = pyemu.geostats.ExpVario(contribution=1.0, a=5000)
    pp_gs = pyemu.geostats.GeoStruct(variograms=pp_v)
    t_d = setup_interface("freyberg_daily",num_reals=100,full_interface=False,include_constants=True,grid_gs=grid_gs,pp_gs=pp_gs,dir_suffix="aniso2")
    plot_mult(t_d,plt_name="mult_aniso2.pdf")

    t_d = setup_interface("freyberg_daily",num_reals=100,full_interface=False,include_constants=True,dir_suffix="base")
    plot_mult(t_d,plt_name="mult_base.pdf")

    # now combine the three prior ensembles into one

    t_d = "daily_aniso1_cond"
    pst = pyemu.Pst(os.path.join(t_d,"freyberg.pst"))
    
    pe1 = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(t_d,"prior.jcb"))
    t_d = "daily_aniso2_cond"
    pe2 = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(t_d,"prior.jcb"))
    
    t_d = "daily_base_cond"
    pe3 = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(t_d,"prior.jcb"))
    
    df = pd.concat([pe1._df,pe2._df,pe3._df],axis=0)
    df.index = np.arange(df.shape[0],dtype=int)
    
    pyemu.ParameterEnsemble(pst=pst,df=df).to_binary(os.path.join(t_d,"prior_combine.jcb"))
    pst.pestpp_options["ies_par_en"] = "prior_combine.jcb"
    pst.pestpp_options["ies_num_reals"] = 300
    pst.write(os.path.join(t_d,"freyberg.pst"),version=2)
    
    # set the observed values and weights for the equality and inequality observations
    set_obsvals_weights(t_d)
    plot_domain(t_d)
    
    # setup the localizer matrix
    build_localizer(t_d)

    # run PESTPP-IES to condition the realizations
    noptmax = 6
    cond_m_d = "daily_cond_combine_master"
    run(t_d,num_workers=20,num_reals=300,noptmax=noptmax,init_lam=-0.1,m_d=cond_m_d)
    
    #make_kickass_figs()
    processing.plot_results_pub(cond_m_d, pstf="freyberg", log_oe=False,noptmax=noptmax)
    processing.plot_histo_pub(cond_m_d,pstf="freyberg",log_oe=False,noptmax=noptmax)
    processing.plot_histo(cond_m_d, pstf="freyberg", log_oe=False, noptmax=noptmax)
    processing.plot_par_changes(cond_m_d,noptmax=noptmax)
    pdfs = [f for f in os.listdir(".") if f.endswith(".pdf") and "ensemble_stack" not in f]
    for pdf in pdfs:
        new_pdf = pdf.replace(".pdf","ensemble_stack.pdf")
        if os.path.exists(new_pdf):
            os.remove(new_pdf)
        shutil.copy2(pdf,new_pdf)


if __name__ == "__main__":


    #ensemble_stacking_experiment()
    #exit()

    # setup the pest interface for conditioning realizations
    # setup_interface("freyberg_daily",num_reals=300,full_interface=False,include_constants=True)
    cond_t_d = "daily_template_cond"
    
    # # set the observed values and weights for the equality and inequality observations
    # set_obsvals_weights(cond_t_d)
    

    # # setup the localizer matrix
    # build_localizer(cond_t_d)
    
    # # run PESTPP-IES to condition the realizations
    # run(cond_t_d,num_workers=20,num_reals=300,noptmax=6,init_lam=-0.1)
    cond_m_d = "daily_master_cond"
    
    # # now setup a corresponding interface that will actually run MODFLOW
    # setup_interface("freyberg_daily",num_reals=500,full_interface=True,include_constants=True)
    flow_t_d = "daily_template"
    
    # # transfer the prior realizations from the previous conditioning analysis into the MODFLOW interace template
    # # so that we are using identical realizations
    # transfer_pars(os.path.join(cond_m_d,"freyberg.pst"),
    #              os.path.join(cond_m_d,"freyberg.0.par.jcb"),
    #              flow_t_d,"cond_prior.jcb")
    
    # # run MODFLOW for the prior realizations
    # run(flow_t_d,num_workers=8,num_reals=100,noptmax=-1,m_d="master_flow_prior")
    

    # now transfer the conditioned realizations into the modflow interface
    transfer_pars(os.path.join(cond_m_d,"freyberg.pst"),
                   os.path.join(cond_m_d,"freyberg.6.par.jcb"),
                   flow_t_d,"cond_post.jcb")

    # now run modflow for the conditioned realizations
    run(flow_t_d,num_workers=8,num_reals=100,noptmax=-1,m_d="master_flow_post")
    
    # now make all the figures for the manuscript
    make_kickass_figs()
    plot_mult(cond_t_d)
    plot_domain()
    processing.plot_results_pub(cond_m_d, pstf="freyberg", log_oe=False,noptmax=6)
    processing.plot_histo_pub(cond_m_d,pstf="freyberg",log_oe=False,noptmax=6)
    processing.plot_histo(cond_m_d, pstf="freyberg", log_oe=False, noptmax=6)
    processing.plot_par_changes(cond_m_d)

