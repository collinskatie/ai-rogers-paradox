from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import os

# In: N, phi
# Out: adapted, is_social_learner, ai_bias, age
@jit(nopython=True)
def create_population(N, phi):
    adapted = np.zeros(N, dtype=np.bool_)
    is_social_learner = np.ones(N, dtype=np.bool_)
    ai_bias = np.full(N, phi)
    age = np.ones(N, dtype=np.int64)
    individual_penalty = np.ones(N, dtype=np.float64)
    return adapted, is_social_learner, ai_bias, age, individual_penalty

# In: N, n_generations, u, c_I, c_AI, z, s0, s1, phi, epsilon_I, mu, n_records, social_learning_mode['', 'human', 'ai', 'both']
# Out: social_learner_freqs, ai_bias_means, change_points, ai_adaptation, learner_adaptation, social_learner_adaptation
@jit(nopython=True)
def run_simulation(N, n_generations, u, c_I, c_AI, z, s0, s1, phi, epsilon_I, mu, n_records, 
                   social_learning_mode="ai", resignation=False, resignation_hint=1, critical=True, ind_penalty_mult=1., learn_twice=False,
                    ai_individ_learn=False, c_AI_i=0.2, z_AI_i=0.9):
    # Initialize population
    adapted, is_social_learner, ai_bias, age, individual_penalty = create_population(N, phi)
    old_ai_adapted = 0.0
    ai_adapted = 0.0
    
    # Metrics
    social_learner_freqs = np.zeros(n_records)
    ai_bias_means = np.zeros(n_records)
    ai_adaptation = np.zeros(n_records)
    learner_adaptation = np.zeros(n_records)
    social_learner_adaptation = np.zeros(n_records)
    change_points = np.zeros(n_records, dtype=np.bool_)
    if critical: assert(social_learning_mode)
    for gen in range(n_generations):
        # Environmental change
        if np.random.random() < u:
            adapted.fill(False)
            ai_adapted = 0.0
            changed = True
        else:
            changed = False
            
        # Learning phase
        new_adapted = np.zeros_like(adapted)
        for i in range(N):
            
            for t in range(1+learn_twice):
                #Do learning loop up to 2 times
                if social_learning_mode and (critical or is_social_learner[i]):# and not resignation)) or (((ai_adapted) > ((1-c_I)*individual_penalty[i]*z)) and resignation):
                    # Social learning (if social learning is enabled and if agents are critical or regular social learners, or if resignation mode is on and E[AI]>E[I])
                    if social_learning_mode == "both":
                        p_copy_ai = ai_bias[i] / (ai_bias[i] + 1)
                    elif social_learning_mode == "ai":
                        p_copy_ai=1.1 
                    elif social_learning_mode == "human":
                        p_copy_ai=-1.
                    if  (((ai_adapted) > ((1-c_I)*individual_penalty[i]*z)) and resignation) or (np.random.random() < p_copy_ai):
                        # Learning from AI if AI has not resigned
                        if np.random.random() < ai_adapted:
                            new_adapted[i] = np.random.random() > epsilon_I
                        individual_penalty[i]=ind_penalty_mult*individual_penalty[i]
                    elif social_learning_mode in ['human','both']:
                        # Learning from random individual
                        teacher = np.random.randint(0, N)
                        if adapted[teacher]:
                            new_adapted[i] = np.random.random() > epsilon_I
                    elif resignation and not ((ai_adapted) > ((1-c_I)*individual_penalty[i]*z)):
                        # Individual learning fallback if AI resigned and no human is available to social learn from 
                        if np.random.random() > c_I*resignation_hint:
                            new_adapted[i] = np.random.random() < (individual_penalty[i]*z)
                else:
                    # Individual learning (if social learning not available, or if agent is neither critical nor regular social learner)
                    if np.random.random() > c_I:
                        new_adapted[i] = np.random.random() < (individual_penalty[i]*z)
                if new_adapted[i]:
                    # Exit learning loop if adapted (short-circuits second loop in learn_twice case if already adapted)
                    break
                if social_learning_mode and critical and not new_adapted[i]:
                    # Individual learning fallback if critical social learner failed at social learning
                    if np.random.random() > c_I:
                        new_adapted[i] = np.random.random() < (individual_penalty[i]*z)
        
        adapted = new_adapted
        
        # Survival
        survival = np.random.random(N) < np.where(adapted, s1, s0)
        adapted = adapted[survival]
        is_social_learner = is_social_learner[survival]
        ai_bias = ai_bias[survival]
        age = age[survival]
        individual_penalty = individual_penalty[survival]
        age += 1
        
        # Reproduction
        current_size = len(adapted)
        n_offspring = N - current_size
        
        if n_offspring > 0:
            # Select parents
            parent_indices = np.random.randint(0, current_size, n_offspring)
            
            # Offspring traits with mutation
            offspring_social = np.zeros(n_offspring, dtype=np.bool)
            offspring_bias = np.zeros(n_offspring)
            
            for j in range(n_offspring):
                # Inherit learning strategy with mutation
                if np.random.random() < mu:
                    offspring_social[j] = not is_social_learner[parent_indices[j]]
                else:
                    offspring_social[j] = is_social_learner[parent_indices[j]]
                
                # Inherit AI bias with mutation
                if np.random.random() < mu:
                    offspring_bias[j] = ai_bias[parent_indices[j]] + np.random.normal(0, 0.1)
                else:
                    offspring_bias[j] = ai_bias[parent_indices[j]]
            
            # Add offspring to population
            adapted = np.append(adapted, np.zeros(n_offspring, dtype=np.bool_))
            is_social_learner = np.append(is_social_learner, offspring_social)
            ai_bias = np.append(ai_bias, offspring_bias)
            age = np.append(age, np.ones(n_offspring, dtype=np.int64))
            individual_penalty = np.append(individual_penalty, np.ones(n_offspring, dtype=np.float64))
        
        
        # AI update
        old_ai_adapted = ai_adapted
        if not ai_individ_learn: 
            # AI can only learn socially (default)
            if np.random.random() > c_AI:
                ai_adapted=np.mean(adapted)
                # ai_adapted = np.quantile(adapted.astype(np.float64),0.50)
        else: 
            if np.random.random() > c_AI_i:
                # Individual learning by AI
                ai_adapted=np.random.random() < z_AI_i
            else: 
                # Social learning from humans
                ai_adapted=np.mean(adapted)
        
        # Record statistics
        if gen >= n_generations - n_records:
            idx = gen - (n_generations - n_records)
            social_learner_freqs[idx] = np.mean(is_social_learner)
            social_mask = is_social_learner
            # if np.any(social_mask):
            ai_bias_means[idx] = np.mean(ai_bias)#[social_mask])
            change_points[idx] = changed
            ai_adaptation[idx]=ai_adapted
            learner_adaptation[idx]=np.mean(adapted)
            
            if np.any(social_mask):
                social_learner_adaptation[idx]=np.mean(adapted[social_mask])
            
    return social_learner_freqs, ai_bias_means, change_points, ai_adaptation, learner_adaptation, social_learner_adaptation

