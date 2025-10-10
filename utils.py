from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd 

def plot_results(params_dict, social_learner_freqs, ai_bias_means, change_points, ai_adaptation, learner_adaptation, 
                 social_learner_adaptation,
                 axis_font_size=18, axis_tick_size=10,
                 burnin=500,
                 time_thresh=-1):
    # Set global font to serif
    plt.rcParams['font.family'] = 'serif'
    
    dname = f"figures/{params_dict['sim_name']}/"
    os.makedirs(dname, exist_ok=1)
    inds = np.where(change_points)
    
    if params_dict["social_learning_mode"] and not params_dict["critical"]:
        plt.figure()
        plt.plot(np.arange(params_dict['n_records'])[burnin:], social_learner_freqs[burnin:])
        plt.ylim((0,1))
        plt.scatter(inds, np.array(change_points)[inds]-1, color="grey")
        plt.ylabel("Social Learner Freq.", fontfamily='serif', fontsize=axis_font_size)
        plt.xlabel("Time", fontfamily='serif', fontsize=axis_font_size)
        plt.xticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.yticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.savefig(dname+f"{params_dict['sim_name']}-social_learner_freq.png")
        #plt.close()

        plt.figure()
        plt.plot(np.arange(params_dict['n_records'])[burnin:], social_learner_adaptation[burnin:], color="grey")
        plt.ylim((0,1))
        plt.ylabel("Social learner adaptation", fontfamily='serif', fontsize=axis_font_size)
        plt.xlabel("Time", fontfamily='serif', fontsize=axis_font_size)
        plt.xticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.yticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.savefig(dname+f"{params_dict['sim_name']}-social_learner_adap.png")
        #plt.close()

    if params_dict["social_learning_mode"] in ["both"]:
        plt.figure()
        plt.plot(np.arange(params_dict['n_records'])[burnin:], ai_bias_means[burnin:], color="grey")
        plt.plot([0,params_dict['n_records'] - burnin],[1,1],linestyle='--')
        plt.ylabel("AI bias", fontfamily='serif', fontsize=axis_font_size)
        plt.xlabel("Time", fontfamily='serif', fontsize=axis_font_size)
        plt.xticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.yticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.savefig(dname+f"{params_dict['sim_name']}-ai_bias.png")
        #plt.close()

        plt.figure()
        vals = ai_bias_means/(ai_bias_means+1)
        plt.plot(np.arange(params_dict['n_records'])[burnin:], vals[burnin:], color="grey")
        plt.plot([0,params_dict['n_records'] - burnin],[1,1],linestyle='--')
        plt.ylabel("AI probability", fontfamily='serif', fontsize=axis_font_size)
        plt.xlabel("Time", fontfamily='serif', fontsize=axis_font_size)
        plt.xticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.yticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.savefig(dname+f"{params_dict['sim_name']}-ai_prob.png")
        #plt.close()

    if params_dict["social_learning_mode"] in ["ai", "both"]:
        
        plt.figure()
        plt.plot(np.arange(params_dict['n_records'])[burnin:], ai_adaptation[burnin:], color="grey")
        plt.ylim((0,1))
        plt.ylabel("AI adaptation", fontfamily='serif', fontsize=axis_font_size)
        plt.xlabel("Time", fontfamily='serif', fontsize=axis_font_size)
        plt.xticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.yticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.savefig(dname+f"{params_dict['sim_name']}-ai_adap.png")
        #plt.close()
    
    print('collective model understanding')
    plt.figure()
    plt.plot(np.arange(params_dict['n_records'])[burnin:], learner_adaptation[burnin:], color="grey")
    plt.ylim((0,1))
    plt.ylabel("Pop World Understanding", fontfamily='serif', fontsize=axis_font_size)
    plt.xlabel("Time", fontfamily='serif', fontsize=axis_font_size)
    plt.xticks(fontfamily='serif', fontsize=axis_tick_size, rotation=70)
    plt.yticks(fontfamily='serif', fontsize=axis_tick_size, rotation=70)
    plt.tight_layout()
    plt.savefig(dname+f"{params_dict['sim_name']}-population_adap.pdf", dpi=300)
    #plt.close()
    
    

def bootstrap_ci(data, n_bootstrap=1000):
    bootstrap_means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means,97.5)
    return lower, upper
    
def get_heatmap(params_dict, all_scores, xvals, yvals, xlabel="", ylabel="", ax_font_size=18, plt_tag="heatmap",
                vmin=0, vmax=1): 
    
    plt.rcParams['font.family'] = 'serif'
    
    dname = f"figures/{params_dict['sim_name']}/"
    os.makedirs(dname, exist_ok=1)
    
    
    scores_array = np.array(all_scores)

    plt.figure(figsize=(10, 8))
    sns.heatmap(scores_array, 
                xticklabels=np.round(xvals, 2),
                yticklabels=np.round(yvals, 2),
                cmap='viridis',
                annot=True,
                vmin=vmin, 
                vmax=vmax,
                fmt='.2f',)
                #cbar_kws={'label': 'Mean Learner Adaptation'})

    # Set labels and title
    plt.xlabel(xlabel, fontsize=ax_font_size)
    plt.ylabel(ylabel, fontsize=ax_font_size)

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(dname+f"{params_dict['sim_name']}-{plt_tag}.pdf", dpi=300)