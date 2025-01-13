from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_results(params_dict, social_learner_freqs, ai_bias_means, change_points, ai_adaptation, learner_adaptation, social_learner_adaptation):
    dname=f"figures/{params_dict['sim_name']}/"
    os.makedirs(dname, exist_ok=1)
    
    inds=np.where(change_points)
    
    if params_dict["social_learning_mode"] and not params_dict["critical"]:
        plt.plot(range(params_dict['n_records']), social_learner_freqs)
        plt.ylim((0,1))
        plt.scatter(inds, np.array(change_points)[inds]-1)
        plt.ylabel("Social Learner Freq.")
        plt.savefig(dname+f"{params_dict['sim_name']}-social_learner_freq.png")
        plt.show()
        
        plt.plot(range(params_dict['n_records']), social_learner_adaptation)
        plt.ylim((0,1))
        # plt.plot([0,params_dict['n_records']],[1,1],linestyle='--')
        plt.ylabel("Social learner adaptation")
        plt.savefig(dname+f"{params_dict['sim_name']}-social_learner_adap.png")
        plt.show()
    
    if params_dict["social_learning_mode"] in ["both"]:
        plt.plot(range(params_dict['n_records']), ai_bias_means)
        plt.plot([0,params_dict['n_records']],[1,1],linestyle='--')
        plt.ylabel("AI bias")
        plt.savefig(dname+f"{params_dict['sim_name']}-ai_bias.png")
        plt.show()
        
        plt.plot(range(params_dict['n_records']), ai_bias_means/(ai_bias_means+1))
        plt.plot([0,params_dict['n_records']],[1,1],linestyle='--')
        plt.ylabel("AI probability")
        plt.savefig(dname+f"{params_dict['sim_name']}-ai_prob.png")
        plt.show()
    if params_dict["social_learning_mode"] in ["ai", "both"]:   
        plt.plot(range(params_dict['n_records']), ai_adaptation)
        plt.ylim((0,1))
        # plt.plot([0,params_dict['n_records']],[1,1],linestyle='--')
        plt.ylabel("AI adaptation")
        plt.savefig(dname+f"{params_dict['sim_name']}-ai_adap.png")
        plt.show()
        
    plt.plot(range(params_dict['n_records']), learner_adaptation)
    plt.ylim((0,1))
    # plt.plot([0,params_dict['n_records']],[1,1],linestyle='--')
    plt.ylabel("Population adaptation")
    plt.savefig(dname+f"{params_dict['sim_name']}-population_adap.png")
    plt.show()
    
    
    # window_size = 1
    # averaged_adaptation = np.convolve(social_learner_adaptation, np.ones(window_size)/window_size, mode='valid')
    # plt.plot(list(range(params_dict['n_records']))[window_size-1:], averaged_adaptation)
    # plt.ylim((0,1))
    # # plt.plot([0,params_dict['n_records']],[1,1],linestyle='--')
    # plt.ylabel("Social learner adaptation")
    # plt.show()