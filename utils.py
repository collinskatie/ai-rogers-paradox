from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_results(params_dict, social_learner_freqs, ai_bias_means, change_points, ai_adaptation, learner_adaptation, social_learner_adaptation,
                 axis_font_size=18, axis_tick_size=10):
    # Set global font to serif
    plt.rcParams['font.family'] = 'serif'
    
    dname = f"figures/{params_dict['sim_name']}/"
    os.makedirs(dname, exist_ok=1)
    inds = np.where(change_points)
    
    if params_dict["social_learning_mode"] and not params_dict["critical"]:
        plt.figure()
        plt.plot(range(params_dict['n_records']), social_learner_freqs)
        plt.ylim((0,1))
        plt.scatter(inds, np.array(change_points)[inds]-1)
        plt.ylabel("Social Learner Freq.", fontfamily='serif', fontsize=axis_font_size)
        plt.xlabel("Time", fontfamily='serif', fontsize=axis_font_size)
        plt.xticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.yticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.savefig(dname+f"{params_dict['sim_name']}-social_learner_freq.png")
        #plt.close()

        plt.figure()
        plt.plot(range(params_dict['n_records']), social_learner_adaptation)
        plt.ylim((0,1))
        plt.ylabel("Social learner adaptation", fontfamily='serif', fontsize=axis_font_size)
        plt.xlabel("Time", fontfamily='serif', fontsize=axis_font_size)
        plt.xticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.yticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.savefig(dname+f"{params_dict['sim_name']}-social_learner_adap.png")
        #plt.close()

    if params_dict["social_learning_mode"] in ["both"]:
        plt.figure()
        plt.plot(range(params_dict['n_records']), ai_bias_means)
        plt.plot([0,params_dict['n_records']],[1,1],linestyle='--')
        plt.ylabel("AI bias", fontfamily='serif', fontsize=axis_font_size)
        plt.xlabel("Time", fontfamily='serif', fontsize=axis_font_size)
        plt.xticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.yticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.savefig(dname+f"{params_dict['sim_name']}-ai_bias.png")
        #plt.close()

        plt.figure()
        plt.plot(range(params_dict['n_records']), ai_bias_means/(ai_bias_means+1))
        plt.plot([0,params_dict['n_records']],[1,1],linestyle='--')
        plt.ylabel("AI probability", fontfamily='serif', fontsize=axis_font_size)
        plt.xlabel("Time", fontfamily='serif', fontsize=axis_font_size)
        plt.xticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.yticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.savefig(dname+f"{params_dict['sim_name']}-ai_prob.png")
        #plt.close()

    if params_dict["social_learning_mode"] in ["ai", "both"]:
        
        plt.figure()
        plt.plot(range(params_dict['n_records']), ai_adaptation)
        plt.ylim((0,1))
        plt.ylabel("AI adaptation", fontfamily='serif', fontsize=axis_font_size)
        plt.xlabel("Time", fontfamily='serif', fontsize=axis_font_size)
        plt.xticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.yticks(fontfamily='serif', fontsize=axis_tick_size)
        plt.savefig(dname+f"{params_dict['sim_name']}-ai_adap.png")
        #plt.close()
    
    print('collective model understanding')
    plt.figure()
    plt.plot(range(params_dict['n_records']), learner_adaptation)
    plt.ylim((0,1))
    plt.ylabel("Pop World Understanding", fontfamily='serif', fontsize=axis_font_size)
    plt.xlabel("Time", fontfamily='serif', fontsize=axis_font_size)
    plt.xticks(fontfamily='serif', fontsize=axis_tick_size, rotation=70)
    plt.yticks(fontfamily='serif', fontsize=axis_tick_size, rotation=70)
    plt.tight_layout()
    plt.savefig(dname+f"{params_dict['sim_name']}-population_adap.pdf", dpi=300)
    #plt.close()
    
    window_size = 1
    plt.figure()
    averaged_adaptation = np.convolve(social_learner_adaptation, np.ones(window_size)/window_size, mode='valid')
    plt.plot(list(range(params_dict['n_records']))[window_size-1:], averaged_adaptation)
    plt.ylim((0,1))
    # plt.plot([0,params_dict['n_records']],[1,1],linestyle='--')
    plt.ylabel("Social learner adaptation")
    plt.show()