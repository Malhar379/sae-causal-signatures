import numpy as np

def generate_world_a(n_samples, p_a=0.3, p_b_given_a1=0.9, p_b_given_a0=0.1,
                      n_distractors=4, p_distractor=0.3, seed=None):
    rng = np.random.default_rng(seed)
    
    A = rng.binomial(1, p_a, size=n_samples)
    B = np.where(
        A == 1,
        rng.binomial(1, p_b_given_a1, size=n_samples),
        rng.binomial(1, p_b_given_a0, size=n_samples)
    )
    
    distractors = rng.binomial(1, p_distractor, size=(n_samples, n_distractors))
    
    X = np.column_stack([A, B, distractors])
    
    return X


def generate_world_b(n_samples, p_c=0.3, p_a_given_c1=0.94, p_a_given_c0=0.06,
                      p_b_given_c1=0.94, p_b_given_c0=0.06,
                      n_distractors=4, p_distractor=0.3, seed=None):
    rng = np.random.default_rng(seed)
    
    C = rng.binomial(1, p_c, size=n_samples)
    
    A = np.where(
        C == 1,
        rng.binomial(1, p_a_given_c1, size=n_samples),
        rng.binomial(1, p_a_given_c0, size=n_samples)
    )
    B = np.where(
        C == 1,
        rng.binomial(1, p_b_given_c1, size=n_samples),
        rng.binomial(1, p_b_given_c0, size=n_samples)
    )
    
    distractors = rng.binomial(1, p_distractor, size=(n_samples, n_distractors))
    
    X = np.column_stack([A, B, distractors])
    
    return X