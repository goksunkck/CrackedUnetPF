import numpy as np

class CrackParticleFilter:
    def __init__(self, num_particles=6000, initial_state=None, 
                 process_noise_std=[0.1, 0.01, 0.01], measurement_noise_std=1.0):
        """
        SIR Particle Filter for Crack Growth Tracking.
        
        State Vector: [a (crack length), logC (material const), m (exponent)]
        
        Args:
            num_particles (int): Number of particles.
            initial_state (dict): Mean and std for [a, logC, m]. 
                                  e.g., {'a_mean': 10, 'a_std': 1, 'logC_mean': -10, ...}
            process_noise_std (list): Std dev of noise added to [a, logC, m] at each step.
            measurement_noise_std (float): Std dev of measurement noise (R).
        """
        self.N = num_particles
        self.process_noise = np.array(process_noise_std)
        self.R = measurement_noise_std
        
        # Initialize particles
        self.particles = np.zeros((self.N, 3))
        
        if initial_state:
            # Random initialization around priors
            self.particles[:, 0] = np.random.normal(initial_state['a_mean'], initial_state['a_std'], self.N)
            self.particles[:, 1] = np.random.normal(initial_state['logC_mean'], initial_state['logC_std'], self.N)
            self.particles[:, 2] = np.random.normal(initial_state['m_mean'], initial_state['m_std'], self.N)
        else:
            # Default fallback (very rough guess)
            self.particles[:, 0] = np.random.uniform(0, 10, self.N)     # a
            self.particles[:, 1] = np.random.uniform(-25, -5, self.N)   # logC
            self.particles[:, 2] = np.random.uniform(2, 4, self.N)      # m
            
        self.weights = np.ones(self.N) / self.N

    def predict(self, delta_K=None, dK_func=None, cycles=1):
        """
        Propagate particles forward using Paris Law: da/dN = C(dK)^m
        
        Args:
            delta_K (float, optional): Stress intensity factor range.
            dK_func (callable, optional): Function dK(a) -> delta_K.
            cycles (int): Number of cycles elapsed since last step.
        """
        a = self.particles[:, 0]
        logC = self.particles[:, 1]
        m = self.particles[:, 2]
        C = 10**logC # Convert logC back to C
        
        # Calculate dK for each particle if function provided, else use constant/scalar
        if dK_func:
            current_dK = dK_func(a)
        else:
            current_dK = delta_K if delta_K is not None else 10.0 # Default fallback
            
        # Paris Law Integration: a_new = a + C * (dK)^m * dN
        # Note: simplistic Euler integration. For larger steps, might need ODE solver.
        da = C * (current_dK ** m) * cycles
        
        # Update State
        self.particles[:, 0] += da
        
        # Add Process Noise (Random Walk for parameters)
        self.particles[:, 0] += np.random.normal(0, self.process_noise[0], self.N)
        self.particles[:, 1] += np.random.normal(0, self.process_noise[1], self.N)
        self.particles[:, 2] += np.random.normal(0, self.process_noise[2], self.N)
        
        # Constraints (Crack Tip Coordinate > Start of Plate)
        # Assuming plate starts at -82.0
        self.particles[:, 0] = np.maximum(self.particles[:, 0], -82.0)

    def update(self, measurement):
        """
        Update particle weights based on measurement (Crack Length).
        
        Args:
            measurement (float): Measured crack length from CNN.
        """
        a_pred = self.particles[:, 0]
        
        # Euclidean distance likelihood (Gaussian)
        # P(z|x) ~ exp(-(z-a)^2 / 2R^2)
        exponent = -0.5 * ((measurement - a_pred)**2) / (self.R**2)
        likelihood = np.exp(exponent)
        
        # Avoid zero weights
        likelihood += 1.e-300
        
        self.weights *= likelihood
        self.weights /= np.sum(self.weights) # Normalize

    def resample(self):
        """
        Systematic Resampling.
        """
        # Effective Sample Size
        N_eff = 1.0 / np.sum(self.weights**2)
        
        # Resample if ESS is too low (e.g., < N/2)
        if N_eff < self.N / 2:
            indices = self.systematic_resample(self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.N) / self.N
            
    def systematic_resample(self, weights):
        N = len(weights)
        positions = (np.arange(N) + np.random.random()) / N
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def estimate(self):
        """
        Returns the mean and variance of the state.
        """
        # Weighted Mean
        mean = np.average(self.particles, weights=self.weights, axis=0)
        
        # Weighted Variance
        var = np.average((self.particles - mean)**2, weights=self.weights, axis=0)
        
        return mean, var
