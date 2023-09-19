import os
import numpy as np
from freyberg import pyemu
# from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors
from matplotlib import cm
import pandas as pd
from freyberg import flopy
from string import ascii_uppercase


def _get_annotation_colors(res, cmap=None, norm=None):
    if cmap is None:
        cmap = cm.RdBu_r
    elif isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    if norm is None:
        norm = colors.CenteredNorm(0, 10)
    tc = 'k'
    fc = cmap(norm(res))
    lu = (0.299 * fc[0] ** 2 + 0.587 * fc[1] ** 2 + 0.114 * fc[2] ** 2) ** 0.5
    if lu < 0.4:
        tc = 'w'
    return fc, tc


def plot_results(m_d, ardim=None, pstf="test_run", log_oe=True,noptmax=None):
    if ardim is None:
        try:
            sim = flopy.mf6.MFSimulation.load(sim_ws=m_d,
                                              load_only=['DIS'])
            m = sim.get_model("freyberg6")
            nrow = m.modelgrid.nrow
            ncol = m.modelgrid.ncol
            ib = m.modelgrid.idomain
        except:
            nlay, nrow, ncol = (1, 50, 50)
            ib = np.loadtxt(os.path.join("org_zone", "ib_resample.dat"), dtype=int)[np.newaxis]
    else:
        nrow, ncol = ardim
    pst = pyemu.Pst(os.path.join(m_d, pstf+".pst"))
    if noptmax is None:
        noptmax = pst.control_data.noptmax
    try:
        pr_oe = pyemu.ObservationEnsemble.from_binary(
            pst=pst, filename=os.path.join(m_d,"{0}.0.obs.jcb".format(pstf))
        )
    except:
        pr_oe = pd.read_csv(os.path.join(m_d,"{0}.0.obs.csv".format(pstf)),
                            low_memory=False,
                            index_col=0)
    try:
        pt_oe = pyemu.ObservationEnsemble.from_binary(
            pst=pst,
            filename=os.path.join(m_d, "{0}.{1}.obs.jcb".format(
                pstf, noptmax))
        )
    except:
        pt_oe = pd.read_csv(
            os.path.join(m_d, "{0}.{1}.obs.csv".format(
                pstf, noptmax)),
            low_memory=False,
            index_col=0)
    try:
        noise = pyemu.ObservationEnsemble.from_binary(
            pst=pst, filename=os.path.join(m_d, "{0}.obs+noise.jcb".format(pstf)))._df
    except:
        noise = pd.read_csv(
            os.path.join(m_d, "{0}.obs+noise.csv".format(pstf)),
                            low_memory=False,
                            index_col=0)
    if log_oe:  # not already logged, transform now
        pr_oe.loc[:, :] = np.log10(pr_oe.values)
        pt_oe.loc[:, :] = np.log10(pt_oe.values)
        noise = noise.copy()
        noise.loc[:, :] = np.log10(noise.values)
        pst.observation_data.loc[:, "obsval"] = np.log10(
            pst.observation_data.obsval.values)

    obs = pst.observation_data.copy()
    obs = obs.loc[obs.otype=="arr"]
    obs.loc[:, "i"] = obs.i.astype(int)
    obs.loc[:, "j"] = obs.j.astype(int)

    onames = obs.oname.unique()
    onames = [o for o in onames if "dup" not in o]
    onames.sort()
    print(onames)
    with PdfPages(os.path.join(m_d,"results.pdf")) as pdf:

        for oname in onames:
            # cheating to get layer from oname

            k = int(oname[-1]) - 1
            oobs = obs.loc[obs.oname==oname,:].copy()
            nzobs = oobs.loc[oobs.weight > 0]
            dups = obs.loc[obs.oname==oname+"dup",:]
            dups = dups.loc[dups.weight > 0]
            lsobs = nzobs.loc[nzobs.obgnme.str.contains("less")].index
            grobs = nzobs.loc[nzobs.obgnme.str.contains("greater")].index
            lsobs = lsobs.append(dups.loc[dups.obgnme.str.contains("less")].index)
            grobs = grobs.append(dups.loc[dups.obgnme.str.contains("greater")].index)
            nz_arr = np.zeros((nrow,ncol))
            nz_arr[nzobs.i,nzobs.j] = nzobs.obsval
            nz_arr[nz_arr==0] = np.NaN
            # catch to not have too many scatters
            if len(nzobs) < 100 and len(nzobs) > 0:
                scatter = True
            else:
                scatter = False
            # means
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            pr_arr = np.zeros((nrow,ncol))
            pr_arr[oobs.i,oobs.j] = pr_oe.loc[:,oobs.obsnme].mean().values
            pt_arr = np.zeros((nrow, ncol))
            pt_arr[oobs.i, oobs.j] = pt_oe.loc[:, oobs.obsnme].mean().values
            pr_arr[ib[k] <= 0] = np.NaN
            pt_arr[ib[k] <= 0] = np.NaN
            mn, mx = min(np.nanmin(pr_arr), np.nanmin(pt_arr)), max(np.nanmax(pr_arr), np.nanmax(pt_arr))

            cb = axes[0].imshow(pr_arr,vmin=mn,vmax=mx)
            plt.colorbar(cb,ax=axes[0],label="$log_{10}$")
            cb = axes[1].imshow(pt_arr, vmin=mn, vmax=mx)
            plt.colorbar(cb, ax=axes[1],label="$log_{10}$")
            grp = oobs.obgnme.iloc[0]
            axes[0].set_title("prior mean "+namer(grp),loc="left")
            axes[1].set_title("posterior mean " + oname, loc="left")
            if scatter:
                ineqobs = nzobs.loc[nzobs.obgnme.str.contains("greater"),:]
                #axes[0].scatter(nzobs.j,nzobs.i,marker=".",s=500,facecolor="none",edgecolor="r")
                #axes[1].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                axes[0].scatter(ineqobs.j, ineqobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                axes[1].scatter(ineqobs.j, ineqobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                eqobs = nzobs.loc[nzobs.obgnme.apply(lambda x: "greater" not in x),:]
                axes[0].scatter(eqobs.j, eqobs.i, marker="^", s=100, facecolor="none", edgecolor="r")
                axes[1].scatter(eqobs.j, eqobs.i, marker="^", s=100, facecolor="none", edgecolor="r")

                # ens mean max residual for group
                tmp_df = pd.concat([nzobs, dups])
                tmp_df['pr_res'] = pr_oe.loc[:, tmp_df.obsnme].mean() - tmp_df.obsval
                tmp_df.loc[tmp_df.index.isin(lsobs) &
                           (tmp_df.pr_res < 0), "pr_res"] = 0
                tmp_df.loc[tmp_df.index.isin(grobs) &
                           (tmp_df.pr_res > 0), 'pr_res'] = 0
                tmp_df['pt_res'] = pt_oe.loc[:, tmp_df.obsnme].mean() - tmp_df.obsval
                tmp_df.loc[tmp_df.index.isin(lsobs) &
                           (tmp_df.pt_res < 0), 'pt_res'] = 0
                tmp_df.loc[tmp_df.index.isin(grobs) &
                           (tmp_df.pt_res > 0), 'pt_res'] = 0
                # set to common obsname
                tmp_df['obsnme'] = tmp_df.obsnme.str.replace(oname + "dup", oname)
                tmp_df = tmp_df.reset_index(drop=True).groupby('obsnme').agg(
                    {'pr_res': lambda x: max(x, key=abs),
                     'pt_res': lambda x: max(x, key=abs)}
                )
                mx_res = tmp_df[['pr_res', "pt_res"]].abs().values.max()
                norm = colors.CenteredNorm(0, mx_res)
                for i,j,name in zip(nzobs.i,nzobs.j,nzobs.obsnme):
                    val = obs.loc[name, "obsval"]
                    prval = pr_arr[i, j]
                    ptval = pt_arr[i, j]
                    grp = obs.loc[name,"obgnme"]
                    dname = name.replace(oname,oname+"dup")
                    if dname in obs.index and obs.loc[dname,"weight"] > 0:
                        lb = "{0:2.2f}".format(obs.loc[dname,"obsval"])
                        ub = "{0:2.2f}".format(nzobs.loc[name, "obsval"])
                        valstr = "{0}<>{1}".format(lb, ub)
                    else:
                        valstr = "{0:2.2f}".format(val)
                        if "less" in grp:
                            valstr = "<" + valstr
                        elif "great" in grp:
                            valstr = ">" + valstr
                    fc, tc = _get_annotation_colors(tmp_df.pr_res[name], norm=norm)
                    axes[0].annotate(
                        "{0}\n{1:2.2f}".format(valstr, prval), (j, i),
                        xytext=(5, 5), textcoords="offset points",
                        va="bottom", ha="left", color=tc,
                        bbox=dict(boxstyle='round', facecolor=fc,
                                  alpha=1.0)
                    )
                    fc, tc = _get_annotation_colors(tmp_df.pt_res[name], norm=norm)
                    axes[1].annotate(
                        "{0}\n{1:2.2f}".format(valstr, ptval), (j, i),
                        xytext=(5, 5), textcoords="offset points",
                        va="bottom", ha="left", color=tc,
                        bbox=dict(boxstyle='round',
                                  facecolor=fc, alpha=1.0)
                    )
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

            # stdev
            nz_arr = np.zeros((nrow, ncol))
            nz_arr[nzobs.i, nzobs.j] = 1. / nzobs.weight
            nz_arr[nz_arr == 0] = np.NaN

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            pr_arr = np.zeros((nrow, ncol))
            pr_arr[oobs.i, oobs.j] = pr_oe.loc[:, oobs.obsnme].std().values
            pt_arr = np.zeros((nrow, ncol))
            pt_arr[oobs.i, oobs.j] = pt_oe.loc[:, oobs.obsnme].std().values
            pr_arr[ib[k] <= 0] = np.NaN
            pt_arr[ib[k] <= 0] = np.NaN
            mn, mx = min(np.nanmin(pr_arr), np.nanmin(pt_arr)), max(np.nanmax(pr_arr), np.nanmax(pt_arr))
            cb = axes[0].imshow(pr_arr, vmin=mn, vmax=mx)
            plt.colorbar(cb, ax=axes[0],label="$log_{10}$")
            cb = axes[1].imshow(pt_arr, vmin=mn, vmax=mx)
            plt.colorbar(cb, ax=axes[1],label="$log_{10}$")
            axes[0].set_title("prior stdev " + oname, loc="left")
            axes[1].set_title("posterior stdev " + oname, loc="left")
            #for ax in axes.flatten():
                #ax.imshow(nz_arr, vmin=mn, vmax=mx)
            if scatter:
                #axes[0].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                #axes[1].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                ineqobs = nzobs.loc[nzobs.obgnme.str.contains("greater"), :]
                # axes[0].scatter(nzobs.j,nzobs.i,marker=".",s=500,facecolor="none",edgecolor="r")
                # axes[1].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                axes[0].scatter(ineqobs.j, ineqobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                axes[1].scatter(ineqobs.j, ineqobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                eqobs = nzobs.loc[nzobs.obgnme.apply(lambda x: "greater" not in x), :]
                axes[0].scatter(eqobs.j, eqobs.i, marker="^", s=100, facecolor="none", edgecolor="r")
                axes[1].scatter(eqobs.j, eqobs.i, marker="^", s=100, facecolor="none", edgecolor="r")

                # ens max std for group
                mx_res = np.max([pr_oe.loc[:, nzobs.obsnme].std(),
                                 pt_oe.loc[:, nzobs.obsnme].std()])
                norm = colors.Normalize(0, mx_res)
                for i,j,name in zip(nzobs.i,nzobs.j,nzobs.obsnme):
                    prval = pr_arr[i, j]
                    ptval = pt_arr[i, j]
                    val = noise.loc[:, name].std()
                    fc, tc = _get_annotation_colors(prval, cmap='BuGn', norm=norm)
                    axes[0].annotate("{0:2.2f}\n{1:2.2f}".format(val, prval), (j, i), xytext=(5, 5),
                                textcoords="offset points", va="bottom", ha="left", color=tc,
                                bbox=dict(boxstyle='round', facecolor=fc, alpha=1.0))
                    fc, tc = _get_annotation_colors(ptval, cmap='BuGn', norm=norm)
                    axes[1].annotate("{0:2.2f}\n{1:2.2f}".format(val, ptval), (j, i), xytext=(5, 5),
                                     textcoords="offset points", va="bottom", ha="left", color=tc,
                                     bbox=dict(boxstyle='round', facecolor=fc, alpha=1.0))

            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

            # min
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            pr_arr = np.zeros((nrow, ncol))
            pr_arr[oobs.i, oobs.j] = pr_oe.loc[:, oobs.obsnme].min().values
            pt_arr = np.zeros((nrow, ncol))
            pt_arr[oobs.i, oobs.j] = pt_oe.loc[:, oobs.obsnme].min().values
            pr_arr[ib[k] <= 0] = np.NaN
            pt_arr[ib[k] <= 0] = np.NaN
            mn, mx = min(np.nanmin(pr_arr), np.nanmin(pt_arr)), max(np.nanmax(pr_arr), np.nanmax(pt_arr))
            cb = axes[0].imshow(pr_arr, vmin=mn, vmax=mx)
            plt.colorbar(cb, ax=axes[0], label="$log_{10}$")
            cb = axes[1].imshow(pt_arr, vmin=mn, vmax=mx)
            plt.colorbar(cb, ax=axes[1], label="$log_{10}$")
            axes[0].set_title("prior min " + oname, loc="left")
            axes[1].set_title("posterior min " + oname, loc="left")
            # for ax in axes.flatten():
            # ax.imshow(nz_arr,vmin=mn,vmax=mx)
            if scatter:
                axes[0].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                axes[1].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")

                # ens max max residual for group
                tmp_df = pd.concat([nzobs, dups])
                tmp_df['obsval'] = noise.loc[:, tmp_df.obsnme].min()
                tmp_df['pr_res'] = pr_oe.loc[:, tmp_df.obsnme].min() - tmp_df.obsval
                tmp_df.loc[tmp_df.index.isin(lsobs) &
                           (tmp_df.pr_res < 0), "pr_res"] = 0
                tmp_df.loc[tmp_df.index.isin(grobs) &
                           (tmp_df.pr_res > 0), 'pr_res'] = 0
                tmp_df['pt_res'] = pt_oe.loc[:, tmp_df.obsnme].min() - tmp_df.obsval
                tmp_df.loc[tmp_df.index.isin(lsobs) &
                           (tmp_df.pt_res < 0), 'pt_res'] = 0
                tmp_df.loc[tmp_df.index.isin(grobs) &
                           (tmp_df.pt_res > 0), 'pt_res'] = 0
                # set to common obsname
                tmp_df['obsnme'] = tmp_df.obsnme.str.replace(oname + "dup", oname)
                tmp_df = tmp_df.reset_index(drop=True).groupby('obsnme').agg(
                    {'pr_res': lambda x: max(x, key=abs),
                     'pt_res': lambda x: max(x, key=abs)}
                )
                mx_res = tmp_df[['pr_res', "pt_res"]].abs().values.max()
                norm = colors.CenteredNorm(0, mx_res)
                for i, j, name in zip(nzobs.i, nzobs.j, nzobs.obsnme):
                    val = noise.loc[:, name].min()
                    prval = pr_arr[i, j]
                    ptval = pt_arr[i, j]
                    grp = obs.loc[name, "obgnme"]
                    dname = name.replace(oname, oname + "dup")
                    if dname in obs.index and obs.loc[dname, "weight"] > 0:
                        lb = "{0:2.2f}".format(obs.loc[dname, "obsval"])
                        ub = "{0:2.2f}".format(nzobs.loc[name, "obsval"])
                        valstr = "{0}<>{1}".format(lb, ub)
                    else:
                        valstr = "{0:2.2f}".format(val)
                        if "less" in grp:
                            valstr = "<" + valstr
                        elif "great" in grp:
                            valstr = ">" + valstr
                    fc, tc = _get_annotation_colors(tmp_df.pr_res[name], norm=norm)
                    axes[0].annotate("{0}\n{1:2.2f}".format(valstr, prval), (j, i), xytext=(5, 5),
                                     textcoords="offset points", va="bottom", ha="left", color=tc,
                                     bbox=dict(boxstyle='round', facecolor=fc, alpha=1.0))
                    fc, tc = _get_annotation_colors(tmp_df.pt_res[name], norm=norm)
                    axes[1].annotate("{0}\n{1:2.2f}".format(valstr, ptval), (j, i), xytext=(5, 5),
                                     textcoords="offset points", va="bottom", ha="left", color=tc,
                                     bbox=dict(boxstyle='round', facecolor=fc, alpha=1.0))
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

            # max
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            pr_arr = np.zeros((nrow, ncol))
            pr_arr[oobs.i, oobs.j] = pr_oe.loc[:, oobs.obsnme].max().values
            pt_arr = np.zeros((nrow, ncol))
            pt_arr[oobs.i, oobs.j] = pt_oe.loc[:, oobs.obsnme].max().values
            pr_arr[ib[k] <= 0] = np.NaN
            pt_arr[ib[k] <= 0] = np.NaN
            mn, mx = min(np.nanmin(pr_arr), np.nanmin(pt_arr)), max(np.nanmax(pr_arr), np.nanmax(pt_arr))
            cb = axes[0].imshow(pr_arr, vmin=mn, vmax=mx)
            plt.colorbar(cb, ax=axes[0], label="$log_{10}$")
            cb = axes[1].imshow(pt_arr, vmin=mn, vmax=mx)
            plt.colorbar(cb, ax=axes[1], label="$log_{10}$")
            axes[0].set_title("prior max {0}".format(oname), loc="left")
            axes[1].set_title("posterior max {0}".format(oname), loc="left")
            # for ax in axes.flatten():
            # ax.imshow(nz_arr,vmin=mn,vmax=mx)
            if scatter:
                axes[0].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                axes[1].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")

                # ens max max residual for group
                tmp_df = pd.concat([nzobs, dups])
                tmp_df['obsval'] = noise.loc[:, tmp_df.obsnme].max()
                tmp_df['pr_res'] = pr_oe.loc[:, tmp_df.obsnme].max() - tmp_df.obsval
                tmp_df.loc[tmp_df.index.isin(lsobs) &
                           (tmp_df.pr_res < 0), "pr_res"] = 0
                tmp_df.loc[tmp_df.index.isin(grobs) &
                           (tmp_df.pr_res > 0), 'pr_res'] = 0
                tmp_df['pt_res'] = pt_oe.loc[:, tmp_df.obsnme].max() - tmp_df.obsval
                tmp_df.loc[tmp_df.index.isin(lsobs) &
                           (tmp_df.pt_res < 0), 'pt_res'] = 0
                tmp_df.loc[tmp_df.index.isin(grobs) &
                           (tmp_df.pt_res > 0), 'pt_res'] = 0
                # set to common obsname
                tmp_df['obsnme'] = tmp_df.obsnme.str.replace(oname + "dup", oname)
                tmp_df = tmp_df.reset_index(drop=True).groupby('obsnme').agg(
                    {'pr_res': lambda x: max(x, key=abs),
                     'pt_res': lambda x: max(x, key=abs)}
                )
                mx_res = tmp_df[['pr_res', "pt_res"]].abs().values.max()
                norm = colors.CenteredNorm(0, mx_res)
                for i, j, name in zip(nzobs.i, nzobs.j, nzobs.obsnme):
                    grp = obs.loc[name, "obgnme"]
                    dname = name.replace(oname, oname + "dup")
                    if dname in obs.index and obs.loc[dname, "weight"] > 0:
                        lb = "{0:2.2f}".format(obs.loc[dname, "obsval"])
                        ub = "{0:2.2f}".format(nzobs.loc[name, "obsval"])
                        valstr = "{0}<>{1}".format(lb, ub)
                    else:
                        valstr = "{0:2.2f}".format(noise.loc[:, name].max())
                        if "less" in grp:
                            valstr = "<" + valstr
                        elif "great" in grp:
                            valstr = ">" + valstr
                    fc, tc = _get_annotation_colors(tmp_df.pr_res[name], norm=norm)
                    axes[0].annotate("{0}\n{1:2.2f}".format(valstr, pr_arr[i, j]), (j, i), xytext=(5, 5),
                                     textcoords="offset points", va="bottom", ha="left", color=tc,
                                     bbox=dict(boxstyle='round', facecolor=fc, alpha=1.0))
                    fc, tc = _get_annotation_colors(tmp_df.pt_res[name], norm=norm)
                    axes[1].annotate("{0}\n{1:2.2f}".format(valstr, pt_arr[i, j]), (j, i), xytext=(5, 5),
                                     textcoords="offset points", va="bottom", ha="left", color=tc,
                                     bbox=dict(boxstyle='round', facecolor=fc, alpha=1.0))
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

            # a few reals
            for idx in pt_oe.index.values[:10]:
                #nz_arr = np.zeros((nrow, ncol))
                #nz_arr[nzobs.i, nzobs.j] = noise.loc[idx,nzobs.obsnme].values
                #nz_arr[nz_arr == 0] = np.NaN
                # means
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                pr_arr = np.zeros((nrow, ncol))
                pr_arr[oobs.i, oobs.j] = pr_oe.loc[idx, oobs.obsnme].values
                pt_arr = np.zeros((nrow, ncol))
                pt_arr[oobs.i, oobs.j] = pt_oe.loc[idx, oobs.obsnme].values
                pr_arr[ib[k] <= 0] = np.NaN
                pt_arr[ib[k] <= 0] = np.NaN
                mn, mx = min(np.nanmin(pr_arr), np.nanmin(pt_arr)), max(np.nanmax(pr_arr), np.nanmax(pt_arr))
                cb = axes[0].imshow(pr_arr, vmin=mn, vmax=mx)
                plt.colorbar(cb, ax=axes[0],label="$log_{10}$")
                cb = axes[1].imshow(pt_arr, vmin=mn, vmax=mx)
                plt.colorbar(cb, ax=axes[1],label="$log_{10}$")
                axes[0].set_title("prior realization {0} {1} mn:{3:3.3f} std:{2:3.3f}".format(idx,oname,np.nanstd(pr_arr),np.nanmean(pr_arr)), loc="left")
                axes[1].set_title("posterior realization {0} {1} mn:{3:3.3f} std:{2:3.3f} ".format(idx,oname,np.nanstd(pt_arr),np.nanmean(pt_arr)), loc="left")
                #for ax in axes.flatten():
                    # ax.imshow(nz_arr,vmin=mn,vmax=mx)
                if scatter:
                    # axes[0].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                    # axes[1].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                    ineqobs = nzobs.loc[nzobs.obgnme.str.contains("greater"), :]
                    # axes[0].scatter(nzobs.j,nzobs.i,marker=".",s=500,facecolor="none",edgecolor="r")
                    # axes[1].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                    axes[0].scatter(ineqobs.j, ineqobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                    axes[1].scatter(ineqobs.j, ineqobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                    eqobs = nzobs.loc[nzobs.obgnme.apply(lambda x: "greater" not in x), :]
                    axes[0].scatter(eqobs.j, eqobs.i, marker="^", s=100, facecolor="none", edgecolor="r")
                    axes[1].scatter(eqobs.j, eqobs.i, marker="^", s=100, facecolor="none", edgecolor="r")

                    # ens max residual for group
                    tmp_df = pd.concat([nzobs, dups])
                    tmp_df['obsval'] = noise.loc[idx, tmp_df.obsnme]
                    tmp_df['pr_res'] = pr_oe.loc[idx, tmp_df.obsnme] - tmp_df.obsval
                    tmp_df.loc[tmp_df.index.isin(lsobs) &
                               (tmp_df.pr_res < 0), "pr_res"] = 0
                    tmp_df.loc[tmp_df.index.isin(grobs) &
                               (tmp_df.pr_res > 0), 'pr_res'] = 0
                    tmp_df['pt_res'] = pt_oe.loc[idx, tmp_df.obsnme] - tmp_df.obsval
                    tmp_df.loc[tmp_df.index.isin(lsobs) &
                               (tmp_df.pt_res < 0), 'pt_res'] = 0
                    tmp_df.loc[tmp_df.index.isin(grobs) &
                               (tmp_df.pt_res > 0), 'pt_res'] = 0
                    # set to common obsname
                    tmp_df['obsnme'] = tmp_df.obsnme.str.replace(oname + "dup", oname)
                    tmp_df = tmp_df.reset_index(drop=True).groupby('obsnme').agg(
                        {'pr_res': lambda x: max(x, key=abs),
                         'pt_res': lambda x: max(x, key=abs)}
                    )
                    mx_res = tmp_df[['pr_res', "pt_res"]].abs().values.max()
                    norm = colors.CenteredNorm(0, mx_res)
                    for i, j, name in zip(nzobs.i, nzobs.j, nzobs.obsnme):
                        grp = obs.loc[name, "obgnme"]
                        dname = name.replace(oname, oname + "dup")
                        if dname in obs.index and obs.loc[dname, "weight"] > 0:
                            lb = "{0:2.2f}".format(obs.loc[dname, "obsval"])
                            ub = "{0:2.2f}".format(nzobs.loc[name, "obsval"])
                            valstr = "{0}<>{1}".format(lb, ub)
                        else:
                            valstr = "{0:2.2f}".format(noise.loc[idx, name])
                            if "less" in grp:
                                valstr = "<{0:2.2f}".format(obs.loc[name, "obsval"])
                            elif "great" in grp:
                                valstr = ">{0:2.2f}".format(obs.loc[name, "obsval"])
                        fc, tc = _get_annotation_colors(tmp_df.pr_res[name], norm=norm)
                        axes[0].annotate("{0}\n{1:2.2f}".format(valstr, pr_arr[i,j]), (j, i), xytext=(5, 5),
                                    textcoords="offset points", va="bottom", ha="left", color=tc,
                                    bbox=dict(boxstyle='round', facecolor=fc, alpha=1.0))
                        fc, tc = _get_annotation_colors(tmp_df.pt_res[name], norm=norm)
                        axes[1].annotate("{0}\n{1:2.2f}".format(valstr, pt_arr[i, j]), (j, i), xytext=(5, 5),
                                         textcoords="offset points", va="bottom", ha="left", color=tc,
                                         bbox=dict(boxstyle='round', facecolor=fc, alpha=1.0))
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)


def namer(name):
    raw = name.split("_")[0]
    print(raw)
    if ":" in raw:
        raw = raw.split(":")[1]
    try:

        layer = int(raw[-1])
    except Exception as e:
        print(name)
        raise Exception("error casting layer: "+str(e))
    tag = raw.split("layer")[0]
    full_name = None
    if tag == "npfk":
        #full_name = "horizontal hydraulic conductivity"
        full_name = "hk"
    elif tag == "stoss":
        #full_name = "specific storage"
        full_name = "ss"
    elif tag == "npfk33":
        #full_name = "vertical hydraulic conductivity"
        full_name = "vk"
    elif tag == "stosy":
        #full_name = "specific yield"
        full_name = "sy"
    else:
        raise Exception("{0}:{1}".format(tag,name))
    full_name += ", layer:{0}".format(layer)
    if "i:" in name:
        row = int(name.split("i:")[1].split("_")[0]) + 1
        col = int(name.split("j:")[1]) + 1
        full_name += ", row:{0}, column:{1}".format(row,col)
    return full_name


def plot_histo(m_d, pstf="test_run", log_oe=True,noptmax=None):

    pst = pyemu.Pst(os.path.join(m_d, pstf+".pst"))
    if noptmax is None:
        noptmax = pst.control_data.noptmax
    pr_oe = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m_d, "{0}.0.obs.jcb".format(pstf)))
    pt_oe = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m_d, "{0}.{1}.obs.jcb". \
                                                                                 format(pstf,noptmax)))
    noise = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m_d, "{0}.obs+noise.jcb".format(pstf)))


    #pr_oe.loc[:, :] = np.log10(pr_oe.values)
    #pt_oe.loc[:, :] = np.log10(pt_oe.values)
    log_noise = noise.copy()
    #log_noise.loc[:, :] = np.log10(log_noise.values)
    #pst.observation_data.loc[:, "obsval"] = np.log10(pst.observation_data.obsval.values)

    obs = pst.observation_data.copy()
    obs = obs.loc[obs.otype == "arr"]
    obs.loc[:, "i"] = obs.i.astype(int)
    obs.loc[:, "j"] = obs.j.astype(int)
    nz_names = obs.loc[obs.weight>0,"obsnme"].to_list()
    with PdfPages(os.path.join(m_d,"histo_results_si.pdf")) as pdf:
        for name in obs.loc[nz_names].sort_values(['i', 'j']).index:
            if "dup" in name:
                continue
            print(name)

            grp = obs.loc[name, "obgnme"]
            fig,ax = plt.subplots(1,1,figsize=(6,3))
            ax.hist(pr_oe.loc[:,name],fc="0.5",ec="none",alpha=0.3,density=True)
            ax.hist(pt_oe.loc[:, name], fc="b", ec="none", alpha=0.3,density=True)
            if log_noise.loc[:,name].var() > 1e-12 and not "less" in grp and not "great" in grp:
                ax.hist(log_noise.loc[:,name],fc="r",ec="none",alpha=0.3,density=True)
            v = obs.loc[name,"obsval"]
            ylim = ax.get_ylim()
            ax.plot([v,v],ylim,"r-",lw=2)
            dname = name.replace("oname:","oname:dup-")
            tag = ""
            if dname in pst.nnz_obs_names:
                v = obs.loc[dname, "obsval"]
                ylim = ax.get_ylim()
                ax.plot([v, v], ylim, "r-", lw=2)
                tag = ", double inequality"
            ax.set_yticklabels([])
            i,j = obs.loc[name,"i"],obs.loc[name,"j"]


            if len(tag) == 0 and "less" in grp:
                tag = ", less-than inequality"
            elif len(tag) == 0 and "great" in grp:
                tag = ", greater-than inequality"

            ax.set_title("{0} {1}".format(namer(name)+" "+name,tag),loc="left")
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)


def plot_histo_pub(m_d, pstf="test_run", log_oe=True, noptmax=None,
                   ineqfill=True):

    # names = ["oname:stosslayer3_otype:arr_i:8_j:47","oname:npfklayer1_otype:arr_i:101_j:23","oname:npfklayer3_otype:arr_i:8_j:47","oname:npfklayer3_otype:arr_i:79_j:31"]
    # units = ["$log_{10} \\frac{1}{m}$","$log_{10} \\frac{m}{d}$","$log_{10} \\frac{m}{d}$","$log_{10} \\frac{m}{d}$"]

    names = ["oname:npfklayer3_otype:arr_i:8_j:47", "oname:npfklayer1_otype:arr_i:101_j:23",
             "oname:npfklayer3_otype:arr_i:79_j:31", "oname:stosslayer1_otype:arr_i:29_j:5"]
    units = ["$log_{10} \\frac{m}{d}$", "$log_{10} \\frac{m}{d}$", "$log_{10} \\frac{m}{d}$", "$log_{10} \\frac{1}{m}$"]

    pst = pyemu.Pst(os.path.join(m_d, pstf+".pst"))
    if noptmax is None:
        noptmax = pst.control_data.noptmax
    pr_oe = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m_d, "{0}.0.obs.jcb".format(pstf)))
    pt_oe = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m_d, "{0}.{1}.obs.jcb". \
                                                                                 format(pstf,noptmax)))
    noise = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m_d, "{0}.obs+noise.jcb".format(pstf)))
    pr_oe = pr_oe.loc[pr_oe.index,:]
    noise = noise.loc[pr_oe.index,:]
    #pr_oe.loc[:, :] = np.log10(pr_oe.values)
    #pt_oe.loc[:, :] = np.log10(pt_oe.values)
    log_noise = noise.copy()
    #log_noise.loc[:, :] = np.log10(log_noise.values)
    #pst.observation_data.loc[:, "obsval"] = np.log10(pst.observation_data.obsval.values)

    obs = pst.observation_data.copy()
    obs = obs.loc[obs.otype == "arr"]
    obs.loc[:, "i"] = obs.i.astype(int)
    obs.loc[:, "j"] = obs.j.astype(int)

    fig, axes = plt.subplots(len(names), 1, figsize=(6, 1.5*len(names)))
    ax_count = 0
    #for name in obs.loc[pst.nnz_obs_names].sort_values(['i', 'j']).index:
    for name in names:
        if name not in names:
            continue
        print(name)
        ax = axes[ax_count]

        grp = obs.loc[name, "obgnme"]

        ax.hist(pr_oe.loc[:,name],fc="0.5",ec="none",alpha=0.3,label="prior ensemble",density=True)
        ax.hist(pt_oe.loc[:, name], fc="b", ec="none", alpha=0.3,label="posterior ensemble",density=True)
        if log_noise.loc[:,name].var() > 1e-12 and not "less" in grp and not "great" in grp:
            ax.hist(log_noise.loc[:,name],fc="r",ec="none",alpha=0.3,label="noise ensemble",density=True)
        v = obs.loc[name,"obsval"]
        ylim = ax.get_ylim()
        ax.plot([v,v],ylim,"r-",lw=2,label="datum", zorder=10)
        dname = name.replace("oname:","oname:dup-")
        tag = ""
        if dname in pst.nnz_obs_names:
            v0 = v
            v = obs.loc[dname, "obsval"]
            ax.plot([v, v], ylim, "r-", lw=2)
            tag = ", double inequality"
            if ineqfill:
                xs = (v0, v)
                ax.fill_between(xs, ylim[0], ylim[1], facecolor="r", alpha=0.3, zorder=9)
        ax.set_yticklabels([])
        xlim = ax.get_xlim()

        if len(tag) == 0 and "less" in grp:
            tag += ", less-than inequality"
            if ineqfill:
                xs = (xlim[0], v)
                ax.fill_between(xs, ylim[0], ylim[1], facecolor="r", alpha=0.3, zorder=9)
        elif len(tag) == 0 and "great" in grp:
            tag += ", greater-than inequality"
            if ineqfill:
                xs = (v, xlim[1])
                ax.fill_between(xs, ylim[0], ylim[1], facecolor="r", alpha=0.3, zorder=9)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(units[ax_count])
        ax.set_yticks([])
        ax.set_title("{2}) {0} {1}".format(namer(name),tag,ascii_uppercase[ax_count]),loc="left")
        if ax_count == 0:
            ax.legend()
        ax_count += 1
    plt.tight_layout()
    plt.savefig(os.path.join(m_d,"prop_histo_pub.pdf"))
    plt.close(fig)


def plot_results_pub(m_d, ardim=None, pstf="test_run", log_oe=True,noptmax=None):
    if ardim is None:
        try:
            sim = flopy.mf6.MFSimulation.load(sim_ws=m_d,
                                              load_only=['DIS'])
            m = sim.get_model("freyberg6")
            nrow = m.modelgrid.nrow
            ncol = m.modelgrid.ncol
            ib = m.modelgrid.idomain
        except:
            nlay, nrow, ncol = (1, 50, 50)
            ib = np.loadtxt(os.path.join("org_zone", "ib_resample.dat"), dtype=int)[np.newaxis]
    else:
        nrow, ncol = ardim
    pst = pyemu.Pst(os.path.join(m_d, pstf+".pst"))
    if noptmax is None:
        noptmax = pst.control_data.noptmax
    try:
        pr_oe = pyemu.ObservationEnsemble.from_binary(
            pst=pst, filename=os.path.join(m_d,"{0}.0.obs.jcb".format(pstf))
        )
    except:
        pr_oe = pd.read_csv(os.path.join(m_d,"{0}.0.obs.csv".format(pstf)),
                            low_memory=False,
                            index_col=0)
    try:
        pt_oe = pyemu.ObservationEnsemble.from_binary(
            pst=pst,
            filename=os.path.join(m_d, "{0}.{1}.obs.jcb".format(
                pstf, noptmax))
        )
    except:
        pt_oe = pd.read_csv(
            os.path.join(m_d, "{0}.{1}.obs.csv".format(
                pstf, noptmax)),
            low_memory=False,
            index_col=0)
    try:
        noise = pyemu.ObservationEnsemble.from_binary(
            pst=pst, filename=os.path.join(m_d, "{0}.obs+noise.jcb".format(pstf)))._df
    except:
        noise = pd.read_csv(
            os.path.join(m_d, "{0}.obs+noise.csv".format(pstf)),
                            low_memory=False,
                            index_col=0)
    if log_oe:  # not already logged, transform now
        pr_oe.loc[:, :] = np.log10(pr_oe.values)
        pt_oe.loc[:, :] = np.log10(pt_oe.values)
        noise = noise.copy()
        noise.loc[:, :] = np.log10(noise.values)
        pst.observation_data.loc[:, "obsval"] = np.log10(
            pst.observation_data.obsval.values)

    pr_oe = pr_oe.loc[pt_oe.index,:]
    obs = pst.observation_data.copy()
    obs = obs.loc[obs.otype=="arr"]
    obs.loc[:, "i"] = obs.i.astype(int)
    obs.loc[:, "j"] = obs.j.astype(int)

    onames = obs.oname.unique()
    onames = [o for o in onames if "dup" not in o]
    onames.sort()
    print(onames)
    ax_count = 0
    with PdfPages(os.path.join(m_d,"results_pub_si.pdf")) as pdf:
        for oname in onames:
            fig, axes = plt.subplots(2, 4, figsize=(14,8.4))
            # cheating to get layer from oname
            #if "klayer3" not in oname and "klayer1" not in oname and "sslayer1" not in oname:
            #    continue
            print(oname)
            k = int(oname[-1]) - 1
            oobs = obs.loc[obs.oname==oname,:].copy()

            nzobs = oobs.loc[oobs.weight > 0]
            dups = obs.loc[obs.oname==oname+"dup",:]
            dups = dups.loc[dups.weight > 0]
            lsobs = nzobs.loc[nzobs.obgnme.str.contains("less")].index
            grobs = nzobs.loc[nzobs.obgnme.str.contains("greater")].index
            lsobs = lsobs.append(dups.loc[dups.obgnme.str.contains("less")].index)
            grobs = grobs.append(dups.loc[dups.obgnme.str.contains("greater")].index)
            nz_arr = np.zeros((nrow,ncol))
            nz_arr[nzobs.i.astype(int),nzobs.j.astype(int)] = nzobs.obsval
            nz_arr[nz_arr==0] = np.NaN
            # catch to not have too many scatters
            if len(nzobs) < 100 and len(nzobs) > 0:
                scatter = True
            else:
                scatter = False

            # stdev
            nz_arr = np.zeros((nrow, ncol))
            nz_arr[nzobs.i.astype(int), nzobs.j.astype(int)] = 1. / nzobs.weight
            nz_arr[nz_arr == 0] = np.NaN

            pr_arr = np.zeros((nrow, ncol))
            pr_arr[oobs.i.astype(int), oobs.j.astype(int)] = pr_oe.loc[:, oobs.obsnme].std().values
            pt_arr = np.zeros((nrow, ncol))
            pt_arr[oobs.i.astype(int), oobs.j.astype(int)] = pt_oe.loc[:, oobs.obsnme].std().values
            pr_arr[ib[k] <= 0] = np.NaN
            pt_arr[ib[k] <= 0] = np.NaN
            truth_arr = np.zeros((nrow,ncol))
            truth_arr[oobs.i.astype(int), oobs.j.astype(int)] = oobs.truth_val.values
            truth_arr[ib[k] <= 0] = np.NaN
                
            mn = min(np.nanmin(pr_arr), np.nanmin(pt_arr),np.nanmin(truth_arr)) 
            mx = max(np.nanmax(pr_arr), np.nanmax(pt_arr),np.nanmax(truth_arr))

            irow = 0
            cb = axes[0,irow].imshow(pr_arr, vmin=mn, vmax=mx)
            plt.colorbar(cb, ax=axes[0,irow],label="$log_{10}$")
            cb = axes[1,irow].imshow(pt_arr, vmin=mn, vmax=mx)
            plt.colorbar(cb, ax=axes[1,irow],label="$log_{10}$")
            grp = oobs.obgnme.iloc[0]
            axes[0,irow].set_title("{0}) prior $\\sigma$ ".format(ax_count) + namer(grp), loc="left")
            ax_count += 1
            axes[1,irow].set_title("{0}) posterior $\\sigma$ ".format(ax_count) + namer(grp), loc="left")
            ax_count += 1
            #for ax in axes.flatten():
                #ax.imshow(nz_arr, vmin=mn, vmax=mx)
            if scatter:
                #axes[0].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                #axes[1].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                ineqobs = nzobs.loc[nzobs.obgnme.str.contains("greater"), :]
                # axes[0].scatter(nzobs.j,nzobs.i,marker=".",s=500,facecolor="none",edgecolor="r")
                # axes[1].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                axes[0,irow].scatter(ineqobs.j, ineqobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                axes[1,irow].scatter(ineqobs.j, ineqobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                eqobs = nzobs.loc[nzobs.obgnme.apply(lambda x: "greater" not in x), :]
                axes[0,irow].scatter(eqobs.j, eqobs.i, marker="^", s=100, facecolor="none", edgecolor="r")
                axes[1,irow].scatter(eqobs.j, eqobs.i, marker="^", s=100, facecolor="none", edgecolor="r")

                # ens max std for group
                mx_res = np.max([pr_oe.loc[:, nzobs.obsnme].std(),
                                 pt_oe.loc[:, nzobs.obsnme].std()])
                norm = colors.Normalize(0, mx_res)
                for i,j,name in zip(nzobs.i,nzobs.j,nzobs.obsnme):
                    prval = pr_arr[i, j]
                    ptval = pt_arr[i, j]
                    val = noise.loc[:, name].std()
                    fc, tc = _get_annotation_colors(prval, cmap='BuGn', norm=norm)
                    axes[0,irow].annotate("{0:2.2f}\n{1:2.2f}".format(val, prval), (j, i), xytext=(5, 5),
                                textcoords="offset points", va="bottom", ha="left", color=tc,
                                bbox=dict(boxstyle='round', facecolor=fc, alpha=1.0))
                    fc, tc = _get_annotation_colors(ptval, cmap='BuGn', norm=norm)
                    axes[1,irow].annotate("{0:2.2f}\n{1:2.2f}".format(val, ptval), (j, i), xytext=(5, 5),
                                     textcoords="offset points", va="bottom", ha="left", color=tc,
                                     bbox=dict(boxstyle='round', facecolor=fc, alpha=1.0))


            # a few reals
            irow = 1
            indices = 5,45
            for idx in [pt_oe.index.values[indices[0]],pt_oe.index.values[indices[1]]]:
                print(idx,grp)
                #nz_arr = np.zeros((nrow, ncol))
                #nz_arr[nzobs.i, nzobs.j] = noise.loc[idx,nzobs.obsnme].values
                #nz_arr[nz_arr == 0] = np.NaN
                # means
                pr_arr = np.zeros((nrow, ncol))
                pr_arr[oobs.i.astype(int), oobs.j.astype(int)] = pr_oe.loc[idx, oobs.obsnme].values
                pt_arr = np.zeros((nrow, ncol))
                pt_arr[oobs.i.astype(int), oobs.j.astype(int)] = pt_oe.loc[idx, oobs.obsnme].values
                pr_arr[ib[k] <= 0] = np.NaN
                pt_arr[ib[k] <= 0] = np.NaN
                mn = min(np.nanmin(pr_arr), np.nanmin(pt_arr),np.nanmin(truth_arr)) 
                mx = max(np.nanmax(pr_arr), np.nanmax(pt_arr),np.nanmax(truth_arr))
                cb = axes[0,irow].imshow(pr_arr, vmin=mn, vmax=mx)
                cbar0 = plt.colorbar(cb, ax=axes[0,irow],label="$log_{10}$")
                cb = axes[1,irow].imshow(pt_arr, vmin=mn, vmax=mx)
                cbar1 = plt.colorbar(cb, ax=axes[1,irow],label="$log_{10}$")



                axes[0,irow].set_title("{2}) prior realization {0}\n{1} ".format(idx,namer(grp),ax_count), loc="left")
                ax_count += 1
                axes[1,irow].set_title("{2}) posterior realization {0}\n{1}".format(idx,namer(grp),ax_count), loc="left")
                ax_count += 1

                #for ax in axes.flatten():
                    # ax.imshow(nz_arr,vmin=mn,vmax=mx)
                if scatter:
                    cbar0.ax.zorder = -1
                    cbar1.ax.zorder = -1
                    # axes[0].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                    # axes[1].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                    ineqobs = nzobs.loc[nzobs.obgnme.apply(lambda x: "greater" in x or "less" in x), :]
                    # axes[0].scatter(nzobs.j,nzobs.i,marker=".",s=500,facecolor="none",edgecolor="r")
                    # axes[1].scatter(nzobs.j, nzobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                    axes[0,irow].scatter(ineqobs.j, ineqobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                    axes[1,irow].scatter(ineqobs.j, ineqobs.i, marker=".", s=500, facecolor="none", edgecolor="r")
                    eqobs = nzobs.loc[nzobs.obgnme.apply(lambda x: "greater" not in x and "less" not in x), :]
                    axes[0,irow].scatter(eqobs.j, eqobs.i, marker="^", s=100, facecolor="none", edgecolor="r")
                    axes[1,irow].scatter(eqobs.j, eqobs.i, marker="^", s=100, facecolor="none", edgecolor="r")

                    # ens max residual for group
                    tmp_df = pd.concat([nzobs, dups])
                    tmp_df['obsval'] = noise.loc[idx, tmp_df.obsnme]
                    tmp_df['pr_res'] = pr_oe.loc[idx, tmp_df.obsnme] - tmp_df.obsval
                    tmp_df.loc[tmp_df.index.isin(lsobs) &
                               (tmp_df.pr_res < 0), "pr_res"] = 0
                    tmp_df.loc[tmp_df.index.isin(grobs) &
                               (tmp_df.pr_res > 0), 'pr_res'] = 0
                    tmp_df['pt_res'] = pt_oe.loc[idx, tmp_df.obsnme] - tmp_df.obsval
                    tmp_df.loc[tmp_df.index.isin(lsobs) &
                               (tmp_df.pt_res < 0), 'pt_res'] = 0
                    tmp_df.loc[tmp_df.index.isin(grobs) &
                               (tmp_df.pt_res > 0), 'pt_res'] = 0
                    # set to common obsname
                    tmp_df['obsnme'] = tmp_df.obsnme.str.replace(oname + "dup", oname)
                    tmp_df = tmp_df.reset_index(drop=True).groupby('obsnme').agg(
                        {'pr_res': lambda x: max(x, key=abs),
                         'pt_res': lambda x: max(x, key=abs)}
                    )
                    mx_res = tmp_df[['pr_res', "pt_res"]].abs().values.max()
                    norm = colors.CenteredNorm(0, mx_res)
                    for i, j, name in zip(nzobs.i, nzobs.j, nzobs.obsnme):
                        ggrp = obs.loc[name, "obgnme"]
                        dname = name.replace("oname:", "oname:dup-")
                        #print(name,dname,dname in obs.index)
                        if dname in obs.index and obs.loc[dname, "weight"] > 0:
                            lb = "{0:2.2f}".format(obs.loc[dname, "obsval"])
                            ub = "{0:2.2f}".format(nzobs.loc[name, "obsval"])
                            valstr = "{0}<>{1}".format(lb, ub)
                        else:
                            valstr = "{0:2.2f}".format(noise.loc[idx, name])
                            if "less" in ggrp:
                                valstr = "<{0:2.2f}".format(obs.loc[name, "obsval"])
                            elif "great" in ggrp:
                                valstr = ">{0:2.2f}".format(obs.loc[name, "obsval"])
                        fc, tc = _get_annotation_colors(tmp_df.pr_res[name], norm=norm)
                        axes[0,irow].annotate("{0}\n{1:2.2f}".format(valstr, pr_arr[i,j]), (j, i), xytext=(5, 5),
                                    textcoords="offset points", va="bottom", ha="left", color=tc,
                                    bbox=dict(boxstyle='round', facecolor=fc, alpha=1.0))
                        fc, tc = _get_annotation_colors(tmp_df.pt_res[name], norm=norm)
                        axes[1,irow].annotate("{0}\n{1:2.2f}".format(valstr, pt_arr[i, j]), (j, i), xytext=(5, 5),
                                         textcoords="offset points", va="bottom", ha="left", color=tc,
                                         bbox=dict(boxstyle='round', facecolor=fc, alpha=1.0))
                irow += 1
            cb = axes[0,-1].imshow(truth_arr, vmin=mn, vmax=mx)
            cbar2 = plt.colorbar(cb, ax=axes[0,-1],label="$log_{10}$")
            axes[0,-1].set_title("{0}) truth {1}".format(ax_count,namer(grp)))
            ax_count += 1
            axes[-1,-1].axis("off")
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)


def plot_par_changes(m_d,noptmax=None,include_insample=False):
    pst = pyemu.Pst(os.path.join(m_d,"freyberg.pst"))
    obs = pst.observation_data
    if noptmax is None:
        noptmax = pst.control_data.noptmax
    pr = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"freyberg.0.par.jcb"))
    pt = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=os.path.join(m_d, "freyberg.{0}.par.jcb".format(noptmax)))
    commonidx = pr.index.intersection(pt.index)
    pt = pt.loc[commonidx, :]

    par = pst.parameter_data
    pnames = par.pname.unique()
    pnames.sort()
    
    tdict = {"cn":"layer constant","pp":"pilot points","gr":"grid-scale","bearing":"bearing"}
    fig,axes = plt.subplots(len(pnames),4,figsize=(11.5,8))
    for irow,pname in enumerate(pnames):
        ppar = par.loc[par.pname==pname,:].copy()
        grps = ppar.pargp.unique()
        grps.sort()
        
        for jcol,grp in enumerate(grps):
            gpar = ppar.loc[ppar.pargp==grp,:]
            ax = axes[irow,jcol]
            ax.hist(pr.loc[:,gpar.parnme].apply(np.log10).values.flatten(),facecolor="0.5",edgecolor="none",alpha=0.5,density=True)
            ax.hist(pt.loc[:, gpar.parnme].apply(np.log10).values.flatten(), facecolor="b", edgecolor="none", alpha=0.5,density=True)
            ax.set_title("{0} {1}, {2} parameters".format(namer(pname),tdict[grp.split("_")[-1]],gpar.shape[0]),loc="left")
            ax.set_yticks([])
            #ax.set_xlabel()
    plt.tight_layout()
    plt.savefig(os.path.join(m_d,"par_change.pdf"))
    plt.close(fig)






