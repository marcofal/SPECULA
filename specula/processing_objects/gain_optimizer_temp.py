import numpy as np
from scipy import signal, optimize
from functools import lru_cache
from statsmodels.tsa.ar_model import AutoReg
import sympy as sp


from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.iir_filter_data import IirFilterData
from specula.processing_objects.iir_filter import IirFilter
from specula.data_objects.simul_params import SimulParams
from specula import cpuArray

import seaborn as sns
import matplotlib.pyplot as plt


class GainOptimizerTemp(BaseProcessingObj):
    """
    Gain optimizer for IIR filters based on modal gain optimization (GENDRON 1994).
    This class optimizes the gains of an IIR filter by minimizing the residual variance
    in the closed-loop system using pseudo open-loop measurements.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 iir_filter_data: IirFilterData,
                 low_pass: IirFilter = None,
                 opt_dt: float = 1.0,  # Optimization interval in seconds
                 identification_dt: float = 1.0,  # Identification interval in seconds
                 delay: float = 2.0,   # Loop delay in frames
                 max_gain_factor: float = 0.95,  # Safety factor for maximum gain
                 safety_factor: float = 0.90,    # Additional safety margin
                 max_inc: float = 0.5,           # Maximum gain increment per step
                 limit_inc: bool = True,         # Limit gain increments
                 ngains: int = 20,               # Number of gain values to test
                 running_mean: bool = False,     # Use running mean for PSD
                 verbose: bool = True,          # Verbose output
                 target_device_idx: int = None,
                 precision: int = None,
                 initial_gain: float = 0.1,
                 prediction_horizon: int = 30,
                 n_realizations: int = 10,
                 ):

        super().__init__(target_device_idx=target_device_idx, precision=precision)
        
        
        self.xp.random.seed(42)  # For reproducibility
        self.simul_params = simul_params
        self.iir_filter_data = iir_filter_data
        self.low_pass_data = low_pass.iir_filter_data if low_pass else None

        self.time_step = simul_params.time_step

        self.x = self.xp.nan  # State vector placeholder
        # Convert optimization parameters
        # self.opt_dt = self.seconds_to_t(opt_dt)
        self.opt_dt = opt_dt / self.time_step  # Ensure at least 1 time step
        self.identification_dt = identification_dt / self.time_step  # Ensure at least 1 time step
        self.delay = delay
        self.max_gain_factor = max_gain_factor
        self.safety_factor = safety_factor
        self.max_inc = max_inc
        self.limit_inc = limit_inc
        self.ngains = ngains #TODO CAREFUL not equal to nmodes
        self.running_mean = running_mean
        self.prediction_horizon = prediction_horizon
        self.n_realizations = n_realizations

        # Get number of modes from filter
        self.nmodes = iir_filter_data.nfilter #1 for the test file
        # print(self.nmodes);exit()

        # History storage
        self.time_hist = []
        self.delta_comm_hist = []
        self.comm_hist = []
        self.data_history = []
        self.optical_gain_hist = []
        self.nperseg_psd = None
        self.psd_ol = None
        self.prev_optimized_gain = self.iir_filter_data.gain.copy()

        self.plot_debug = False  # Enable plotting for debugging

        self.A, self.B, self.C, self.D, self.H_cl = self._generate_state_space() #generate state space matrices for the closed loop system
        # self._A_func = sp.lambdify(sp.symbols('g', real=True, positive=True), self.A, modules='numpy')
        # self._B_func = sp.lambdify(sp.symbols('g', real=True, positive=True), self.B, modules='numpy')
        # self._C_func = sp.lambdify(sp.symbols('g', real=True, positive=True), self.C, modules='numpy')
        # self._D_func = sp.lambdify(sp.symbols('g', real=True, positive=True), self.D, modules='numpy')

        
        # self.iir_filter_data.set_gain(self.xp.repeat(initial_gain, self.nmodes))  # Set initial gain
        self.iir_filter_data.set_gain(self.xp.repeat(initial_gain, self.nmodes))  # Set initial gain
        
        self.optimized_gain = BaseValue(
            value=self.xp.ones(self.nmodes, dtype=self.dtype),
            target_device_idx=target_device_idx
        )

        # self.optimized_gain = BaseValue(  #just a scalar for us
        #     value = None,
        #     target_device_idx=target_device_idx
        # )
        self.cost_functions = BaseValue(
            value = None,
            target_device_idx=target_device_idx
        )

        # Initialize optimal gain to ones
        self.optimized_gain.value = self.xp.ones(self.nmodes, dtype=self.dtype)
        # self.optimized_gain.value = 1.0

        self.optimization_done = False

        # # Inputs
        # self.inputs['delta_comm'] = InputValue(type=BaseValue)
        # self.inputs['out_comm'] = InputValue(type=BaseValue)
        # self.inputs['optical_gain'] = InputValue(type=BaseValue, optional=True)
        self.inputs["exo_data"] = InputValue(type=BaseValue, optional=True)  # Exogenous data if needed
        # self.inputs['data_history'] = InputValue(type=BaseValue, optional=False)  # History of tip/tilt measurements
        # Outputs
        self.outputs['optimized_gain'] = self.optimized_gain
        # self.outputs["cost_functions"] = self.cost_functions

        self.verbose = verbose

    def _generate_state_space(self):
        """
        Generate a state-space representation (A, B, C, D) and the symbolic closed-loop transfer
        function H_cl for a single-mode IIR regulator described by iir_filter_data.

        This routine expects the filter coefficients to be provided in ascending powers of z
        (i.e. coefficients for z^0, z^1, ..., z^m). It builds a companion-form discrete-time
        state-space model for the closed-loop path that includes explicit sensor and actuator
        sample delays split from an integer total delay stored on the instance.

        Parameters
        ----------
        iir_filter_data : object
            An object (or struct-like) that exposes at least the attributes:
            - num : array-like
                Numerator coefficients of the IIR regulator given in ascending powers
                (z^0 first).
            - den : array-like
                Denominator coefficients of the IIR regulator given in ascending powers
                (z^0 first).
            The function may also use instance attributes (self.iir_filter_data) if present.
            Coefficients may be numeric (numpy/cupy arrays) or SymPy expressions convertible to
            SymPy objects.

        Notes on instance state
        -----------------------
        - self.delay (int-like) is read to determine the total integer sample delay.
          The delay is split into sensor_delay = total_delay // 2 and
          actuator_delay = total_delay - sensor_delay.
        - Numerical arrays are accessed through the instance numeric backend (self.xp)
          and may be converted to/from a CPU array helper (cpuArray) if present.

        Behavior and algorithm
        ----------------------
        1. Constructs SymPy symbols z (complex) and g (real, positive) and multiplies the
           regulator numerator by the symbolic test gain g so that the returned transfer
           function H_cl depends on g.
        2. Forms the regulator transfer function R(z) = Num(z) / Den(z) using the provided
           coefficients (care taken to respect the ascending-order convention).
        3. Represents pure delays in the z-domain as S(z) = z^-sensor_delay and
           D(z) = z^-actuator_delay.
        4. Forms the closed-loop scalar transfer function from sensor output to controller
           output as:
               H_cl(z) = S(z) / (1 + S(z) * R(z) * D(z))
           and simplifies it to a rational expression using SymPy.
        5. Extracts the numerator and denominator polynomials of H_cl, normalizes the
           denominator to be monic (leading coefficient 1), and constructs a companion
           (controllable canonical) A matrix of size n = deg(den).
        6. Constructs B as a column vector with a 1 in the last entry, C as the row
           vector of numerator coefficients (padded on the left with zeros to length n),
           and sets feedthrough D to 0 (the implementation assumes strict-properness
           of the dynamic part; however numerical/symbolic conversions may leave a
           numerator of degree equal to n — in which case D remains 0 in the returned
           tuple).

        Returns
        -------
        A : sympy.Matrix
            n x n state matrix in companion form (monic denominator assumed).
        B : sympy.Matrix
            n x 1 input vector (zeros except last row = 1).
        C : sympy.Matrix
            1 x n output row vector containing numerator coefficients of H_cl,
            ordered for descending powers of z (padded to length n).
        D : sympy.Expr or sympy.Integer
            Feedthrough term (set to 0).
        H_cl : sympy.Expr
            The symbolic closed-loop transfer function S(z) / (1 + S(z) R(z) D(z)),
            expressed in the SymPy symbol z and parameterized by the symbolic gain g.

        Exceptions and edge cases
        -------------------------
        - If the denominator polynomial of H_cl is constant (degree 0) the companion
          construction will not produce a meaningful A/B/C of size >= 1; callers should
          ensure the regulator produces a proper dynamic transfer function.
        - Coefficient ordering is important: input numerator/denominator arrays must use
          ascending powers (z^0 first). If inputs are given in descending order convert
          them beforehand.
        - The routine relies on SymPy polynomial algebra; very large symbolic expressions
          may be expensive to simplify/manipulate.

        Example
        -------
        Assuming an IIR filter with numerator [b0, b1, b2] and denominator [a0, a1, a2]
        in ascending order, and self.delay = 3, the function will:
        - split delay into sensor_delay = 1, actuator_delay = 2,
        - form R(z) with symbolic gain g,
        - compute H_cl(z) = z^-1 / (1 + z^-1 * R(z) * z^-2),
        - return companion-form (A, B, C, D) and the symbolic H_cl.

        """

        z = sp.symbols('z', complex=True)
        g = sp.symbols('g', real=True, positive=True)

        # Get filter coefficients for this mode from iir_filter_data (assume single-mode for this optimizer)
        # iir_filter_data stores coefficients in ascending powers (z^0 first) so keep that convention for lfilter
        num = self.xp.asarray(cpuArray(self.iir_filter_data.num.copy()[0, :]))
        # num[0] * z**-2 + num[1] * z**-1 + num[2] * z**0
        den = self.xp.asarray(cpuArray(self.iir_filter_data.den.copy()[0, :]))

        # Normalize numerator by the original gain and apply new test gain
        # orig_gain = float(cpuArray(self.iir_filter_data.gain[0]))
        # orig_gain = g
        num_normalized = self.xp.asarray(num, dtype=float) * g
        # Determine integer delays (in samples). Use rounding to allow non-integer attributes.
        total_delay = int(round(self.delay))
        sensor_delay = total_delay // 2
        actuator_delay = total_delay - sensor_delay

        # Calculate regulator transfer function R(z)
        num_val = self.xp.polyval(num_normalized[::-1], z)
        den_val = self.xp.polyval(den[::-1], z)
        # den_val = self.xp.where(self.xp.abs(den_val) < 1e-12, 1e-12, den_val)
        R_tf = num_val / den_val

        # Sensor and actuator delays in z-domain
        S_tf = z**(-sensor_delay)
        D_tf = z**(-actuator_delay)

        #Add the Low-pass mirror dynamics 
        lp_num = self.xp.asarray(cpuArray(self.low_pass_data.num.copy()[0, :])) if self.low_pass_data else self.xp.array([1.0])
        lp_den = self.xp.asarray(cpuArray(self.low_pass_data.den.copy()[0, :])) if self.low_pass_data else self.xp.array([1.0])

        lp_num_val = self.xp.polyval(lp_num[::-1], z)
        lp_den_val = self.xp.polyval(lp_den[::-1], z)

        LP_tf = lp_num_val / lp_den_val

        # Closed-loop transfer function: H_cl = S / (1 + S * R * D)
        open_loop = S_tf * R_tf * D_tf * LP_tf
        denominator = 1.0 + open_loop
        # denominator = self.xp.where(self.xp.abs(denominator) < 1e-12, 1e-12, denominator)
        H_cl = S_tf / denominator
        # Extract and print the coefficients of z in the numerator and denominator of H

        num_expr, den_expr = sp.fraction(sp.together(H_cl))
        num_poly = sp.Poly(num_expr, z)
        den_poly = sp.Poly(den_expr, z)
        # quotient, remainder = sp.div(num_poly.as_expr(), den_poly.as_expr(), z)
        # H_constant = (quotient)
        # H_proper = (remainder/den_poly.as_expr())
        # H = H_constant + H_proper
        # print("H as a proper function plus a constant:")
        # sp.pprint(H_cl)
        # exit()
        # num_expr, den_expr = sp.fraction(H)
        # den_poly = sp.Poly(den_expr, z)
        num_coeffs = num_poly.all_coeffs()  # Coefficients in descending powers of z
        den_coeffs = den_poly.all_coeffs()  # Coefficients in descending powers of z

        # Normalize the denominator (make it monic)
        lead = den_poly.LC()
        den_poly = sp.Poly(den_poly.as_expr() / lead, z)
        num_poly = sp.Poly(num_poly.as_expr() / lead, z)

        print("Normalized numerator polynomial:"
            , num_poly)
        print("Normalized denominator polynomial:"
            , den_poly)

        # Let n be the degree of the denominator
        n = den_poly.degree()
        # Get coefficients of den(z) = z^n + a_{n-1} z^(n-1) + … + a_0;
        # all_coeffs() returns [1, a_{n-1}, …, a_0]
        den_coeffs = den_poly.all_coeffs()
        A = sp.zeros(n)
        for i in range(n - 1):
            A[i, i + 1] = 1
        # Last row: use the den_coeffs in reverse (skip the leading 1)
        A[n - 1, :] = sp.Matrix([-den_coeffs[-(j + 1)] for j in range(n)]).T

        # Input vector B: only the last entry is 1
        B = sp.zeros(n, 1)
        B[n - 1] = 1

        # For a strictly proper system the numerator degree is less than n.
        # Get the coefficients of the numerator (in descending powers).
        num_coeffs = num_poly.all_coeffs()
        # Pad with zeros on the left if needed so that num_coeffs has length n
        if len(num_coeffs) < n:
            num_coeffs = [0] * (n - len(num_coeffs)) + num_coeffs
        # Define the output matrix C as a row vector with these coefficients
        C = sp.Matrix(num_coeffs[::-1]).T  # convertible to a 1xn row vector when needed

        # Since H(z) is strictly proper, we set the feedthrough term D to zero.
        D = sp.sympify(0)

        self.x = self.xp.zeros((n, self.n_realizations), dtype=self.dtype)  # State vector placeholder
        # A, B, C, D = iir_filter_data.to_state_space()
        return A, B, C, D, H_cl
    
    def setup(self):
        super().setup()
        self.build_stream()

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        # Get current inputs
        # self.current_delta_comm = self.local_inputs['delta_comm'].value
        # self.current_out_comm = self.local_inputs['out_comm'].value
        # Maybe retrieve something to perform optimization. Could be --> parameters of the AR, could be history batch
        self.exo_data = self.local_inputs['exo_data'].value #will contain an history of tip/tilt

    def trigger_code(self):
        t = self.current_time
        current_timestep = self.t_to_seconds(t) / self.time_step

        # Store history
        self.time_hist.append(t)
        # print("self.exo_data:", self.exo_data.shape);exit() #is as 2,
        self.data_history.append(self.exo_data[:self.nmodes].copy()) #Consider only the components of distortion that you want to optimize to

        if self.verbose and current_timestep % self.opt_dt == 0:
            # print(self.xp.array(self.data_history).shape)
            print(f"Current time step: {current_timestep}, time: {self.t_to_seconds(t):.3f}s")
            print(f"GainOptimizerCustom: Stored data at t={self.t_to_seconds(t):.3f}s, total stored: {len(self.data_history)}")
            print(f"Exogenous data (first {self.ngains} samples): {self.exo_data[:self.ngains]}")
            print(f"Data history length: {(self.identification_dt)}")
            print(f"Current optimized gain: {self.prev_optimized_gain}")


        # Check if it's time to optimize
        if current_timestep >= self.opt_dt and (current_timestep % self.opt_dt) == 0:
            self._optimize_gain(t)
            self.optimization_done = True
        else:
            self.optimization_done = False

    def _optimize_gain(self, t):
        """
        Perform gain optimization based on accumulated history. Its a SISO optimizer for now
        """

        if len(self.data_history) < self.opt_dt:
            if self.verbose:
                print("Not enough data history for optimization.")
            return

        #Once at least opt_dt samples are stored, perform optimization
        data_to_consider = int(min(self.identification_dt, len(self.data_history)))

        data = self.xp.stack([self.xp.asarray(d[1:2]) for d in self.data_history])[-data_to_consider:, :].flatten()     # Use only the last opt_dt samples of the tilt mode TODO
        # print(data.shape);exit()

        if self.plot_debug:
            # Convert stored history to a NumPy array and plot the last opt_dt samples for each mode

            # Build a (N, nmodes) numpy array from the stored history
            time_plot = self.xp.asarray([self.t_to_seconds(tt) for tt in self.time_hist])

            # Choose how many points to show (use at most opt_dt samples)
            n_plot = min(len(data), int(self.opt_dt))
            if n_plot == 0:
                return

            idx = slice(-n_plot, None)
            t_plot = time_plot[idx]

            plt.figure(figsize=(10, 4 + 1 * self.nmodes))
            # for m in range(self.nmodes):
            plt.plot(t_plot, data[-n_plot:])
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title(f"Data history (last {n_plot} samples)")
            plt.grid(True)
            if self.nmodes <= 10:
                plt.legend(loc="upper right", fontsize="small")
            plt.tight_layout()
            plt.show()
        
        # Calculate maximum stable gains
        gmax_vec = self._calculate_max_gains() 
        max_g = self.xp.max(gmax_vec)

        turb_pred = self._predict_turbulent_signal(data, order=10) #last opt_dt values to learn
        # print(turb_pred.shape);exit()
        opt_g = self._optimize_g(turb_pred, max_g) # predicted signal to optimize  --> minimum energy
        # Ensure opt_g and max_g are xp arrays and clip elementwise.
        # opt_arr = self.xp.asarray(opt_g)
        # max_vec = self.xp.asarray(max_g)

        clipped_optimal_gain = self.xp.minimum(opt_g, max_g).astype(self.dtype)
        
        # #Clip all components of the vector, if it is a vector
        # if opt_arr.ndim == 0 or opt_arr.size == 1:
        #     # scalar optimal gain -> clip every component of max_g to that scalar
        #     clipped_optimal_gain = self.xp.minimum(float(opt_arr), max_vec).astype(self.dtype)
        # else:
        #     # vector optimal gain -> try elementwise clipping, broadcast if shapes differ
        #     try:
        #         clipped_optimal_gain = self.xp.minimum(opt_arr, max_vec).astype(self.dtype)
        #     except Exception:
        #         clipped_optimal_gain = self.xp.minimum(self.xp.broadcast_to(opt_arr, max_vec.shape), max_vec).astype(self.dtype)

        # print(f"Optimal gain: {opt_g}, Clipped optimal gain: {clipped_optimal_gain}, Max stable gain: {max_g}")

        # Store results
        self.prev_optimized_gain = clipped_optimal_gain.copy()
        # self.optimized_gain.value = self.xp.array([clipped_optimal_gain], dtype=self.dtype)  
        self.optimized_gain.value = clipped_optimal_gain

        if self.verbose:
            print(f"Optimized gains at t={self.t_to_seconds(t):.3f}s: "
                  f"optimal_gain = {self.optimized_gain.value:.4f}, ")
                #   f"mean={float(self.xp.mean(clipped_optimal_gain)):.4f}")

    


    def _exo_id_autoreg(self, data,id_hor,order):
        """
        Identification of the exosystem with an autoregressive model of a given order
        """
        model = AutoReg(data[-id_hor:], lags=order, old_names=False)
        model_fit = model.fit()
        ar_params = model_fit.params[1:]  # exclude intercept

        # Build companion matrix A
        A = np.zeros((order, order))
        A[0, :] = ar_params
        for i in range(1, order):
            A[i, i-1] = 1.0

        # Get the noise standard deviation from the model
        noise_std = np.sqrt(model_fit.sigma2)

        
        return A, model_fit, noise_std
    

    def _predict_turbulent_signal(self, data, order):
        """
        predict the turbulent signal in the predictive optimization horizon (opt_dt)
        """
        A_exo, model_fit, noise_std = self._exo_id_autoreg(data, id_hor=self.prediction_horizon, order=order)
        # Get the noise standard deviation from the model

        x = self.xp.zeros((A_exo.shape[0], self.prediction_horizon, self.n_realizations), dtype=self.dtype) #state, time, realization
        y = self.xp.zeros((self.n_realizations, self.prediction_horizon), dtype=self.dtype) #realization, time
        x[:,0,:] = self.xp.repeat(data[-order:].copy().astype(self.dtype), self.n_realizations).reshape(order, self.n_realizations)  # Use the last 'order' values of data
        y[:,0] = self.xp.repeat(data[-1].copy().astype(self.dtype), self.n_realizations).reshape(self.n_realizations)  # First predicted output is the last known data point

        for t in range(1, self.prediction_horizon):
            
            noise = self.xp.zeros((order, self.n_realizations), dtype=self.dtype)
            noise_t = self.xp.random.randn(self.n_realizations) * noise_std
            #create noise_t vector --> order, n_realizations

            noise[0, :] = noise_t
            #

            # print(f"noise_t shape: {noise_t.shape}, noise shape: {noise.shape}, x shape: {x[:,t:t+1, :] .shape}, A_exo shape: {A_exo.shape}, t: {t}")

            # print(f"A_exo shape: {A_exo.shape}, x shape: {x.shape}, noise_t shape: {noise_t.shape}")
            for j in range(self.n_realizations):
                x[:,t, j:j+1] = A_exo @ x[:,t-1, j:j+1] + noise[:, j:j+1]
                y[j, t] = x[0,t, j]

            
        
        if self.plot_debug:

            plt.figure(figsize=(10, 6))
            # Calculate mean and standard deviation across realizations
            y_mean = self.xp.mean(y, axis=0)
            y_std = self.xp.std(y, axis=0)

            # Create time axis for plotting
            t_axis = self.xp.arange(self.prediction_horizon) * self.time_step

            # Convert to CPU arrays for plotting
            t_plot = cpuArray(t_axis)
            y_mean_plot = cpuArray(y_mean)
            y_std_plot = cpuArray(y_std)
            y_plot = cpuArray(y)

            # Plot individual realizations
            for j in range(min(self.n_realizations, 10)):  # Limit to 10 realizations for clarity
                plt.plot(t_plot, y_plot[j, :], alpha=0.3, linewidth=0.5, color='gray')

            # Plot mean signal
            plt.plot(t_plot, y_mean_plot, 'b-', linewidth=2, label='Mean')

            # Plot variance tube (mean ± std)
            plt.fill_between(t_plot, 
                             y_mean_plot - y_std_plot, 
                             y_mean_plot + y_std_plot, 
                             alpha=0.3, color='blue', label='±1 Std Dev')

            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Predicted Turbulent Signal ({self.n_realizations} realizations)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()



        return y

    def _optimize_g(self, predicted_signal, max_gain):
        """
        Optimize gain based on predicted turbulent signal.
        """
        #TODO vector geeralization?
        res = optimize.minimize_scalar(
            lambda gg: self._cost_function(gg, predicted_signal), 
            # bracket=(0.0, self.iir_filter_data.gain.copy()[0], max_gain),
            bounds=(0.0, max_gain), 
            method='bounded', 
            options={"maxiter": 1e7, "xatol": 1e-16}
        )
        # x0 = self.dtype(0.5 * max_gain)
        # max_gn = self.dtype(max_gain)
        # res = optimize.minimize(
        #     # fun=lambda gg: self.dtype(self._cost_function(gg, predicted_signal)),
        #     fun=lambda gg: self.dtype(self._cost_function(gg, predicted_signal)),
        #     x0= self.iir_filter_data.gain.copy()[0],  # Initial guess
        #     # args=(predicted_signal, True),
        #     method="L-BFGS-B",
        #     bounds=[(0.0, max_gn)],
        #     options={"maxiter": int(1e6), "ftol": 1e-16, "gtol": 1e-16},
        # )
        # print(res.x);exit()
        # # # ensure downstream code sees a scalar like minimize_scalar returned
        # if isinstance(res.x, self.dtype):
        #     res.x = self.dtype(res.x.ravel()[0])

        return res.x

    def _cost_function(self, gain, predicted_signal, time_domain: bool = True):

        """
        Cost function to minimize: squared sum of output signal y.
        
        Computes the output y of the closed-loop system with transfer function:
        H_cl = S / (1 + S * R * D)
        
        Where:
        - S: sensor (simple delay)
        - R: regulator (IIR filter with given gain)
        - D: actuator (another delay)
        
        Args:
            gain: The gain to test for the IIR filter
            t: Current time step
            x: Current state
            predicted_signal: Input signal (turbulent prediction)
            
        Returns:
            Cost: squared sum of output signal y
        """
        
        # Extract scalar gain value from array if needed
        if hasattr(gain, '__len__'):
            gain = float(gain[0])
        else:
            gain = float(gain)

        z = sp.symbols('z', complex=True)
        g = sp.symbols('g', real=True, positive=True)

        n_points = predicted_signal.shape[1]
        fs = 1.0 / self.time_step
        freq = self.xp.fft.fftfreq(n_points, 1 / fs)
        omega = 2 * self.xp.pi * freq
        z_subs = self.xp.exp(1j * omega * self.time_step)

        # FFT-based computation (default): apply H_cl in frequency domain
        if not time_domain:
            """
            Perform a wrapped FFT-based computation of the closed-loop output. In the sense that you assume the window of the predicted signal is actually the
            window of a periodic signal, so you can use FFT directly without edge effects. This is reliable if the response is much shorter than the window length
            """

            subs = {g: gain, z: z_subs}
            H_cl = self.H_cl.subs(subs)
            # Clean up any NaN/Inf
            H_cl = self.xp.nan_to_num(H_cl, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply transfer function to predicted signal in frequency domain
            predicted_signal_fft = self.xp.fft.fft(predicted_signal)
            output_fft = H_cl * predicted_signal_fft

            # Convert back to time domain
            y = self.xp.real(self.xp.fft.ifft(output_fft))

            # Compute cost as squared sum of output signal
            total_variance = self.xp.sum(y**2)
            return total_variance

        # Time-domain computation: build H_cl as rational transfer function and use lfilter
        # R(z) = B(z^-1) / A(z^-1) where arrays are in ascending z^-1 powers (b0 + b1 z^-1 + ...)
        # H_cl(z) = S(z) / (1 + S(z) R(z) D(z))
        # With S = z^-sensor_delay, D = z^-actuator_delay and R = B/A:
        # H_cl = A * z^-sensor_delay / (A + z^-total_delay * B)

        A = self.xp.array(self.A.subs(g, gain), dtype=self.dtype)
        B = self.xp.array(self.B.subs(g, gain), dtype=self.dtype)
        C = self.xp.array(self.C.subs(g, gain), dtype=self.dtype)
        D = self.xp.array(self.D.subs(g, gain), dtype=self.dtype)

        # A = self._A_func(gain)
        # B = self._B_func(gain)
        # C = self._C_func(gain)
        # D = self._D_func(gain)

        x = self.x.copy()
        # x = self.xp.zeros((A.shape[0], predicted_signal.shape[1]), dtype=self.dtype) #state, prediction horizon
        # print(x.shape);exit()
        y_td = self.xp.zeros((n_points, self.n_realizations), dtype=self.dtype) #time, realizations

        
        for t in range(n_points):
            for j in range(self.n_realizations):
                x[:, j:j+1] = A @ x[:, j:j+1] + B * predicted_signal[j:j+1,t:t+1]
                y_td[t, j] = (C @ x[:, j:j+1] + D * predicted_signal[j:j+1,t:t+1])
        


        self.x = x.copy()

        return 1/self.n_realizations * self.xp.linalg.norm(y_td)

        
    def _calculate_max_gains(self):
        """
        Calculate maximum stable gains for each mode using IirFilterData stability analysis.
        """
        # Use the new max_stable_gain method from IirFilterData
        gmax_vec = self.iir_filter_data.max_stable_gain(
            delay=self.delay,
            max_gain=1.0,  # Maximum gain to test
            n_gain=20000,   # Number of gain values to test for high precision
        )

        # Apply the maximum gain factor safety margin
        gmax_vec = self.to_xp(gmax_vec) * self.max_gain_factor

        if self.verbose:
            print("Maximum stable gains calculated:")
            print(f"  Raw max gains: mean={float(self.xp.mean(gmax_vec/self.max_gain_factor)):.4f}, "
                f"std={float(self.xp.std(gmax_vec/self.max_gain_factor)):.4f}")
            print(f"  With safety factor ({self.max_gain_factor}): mean={float(self.xp.mean(gmax_vec)):.4f}, "
                f"std={float(self.xp.std(gmax_vec)):.4f}")

        return gmax_vec

    def post_trigger(self):
        super().post_trigger()

        if self.optimization_done:
            self.optimized_gain.generation_time = self.current_time
            # Replicate optimized gain across all modes before setting it
            val = self.optimized_gain.value
            val_arr = self.xp.asarray(val)
            if val_arr.size == 1:
                gain_vec = self.xp.full(self.nmodes, float(val_arr), dtype=self.dtype)
            else:
                gain_vec = val_arr.copy().astype(self.dtype)
                if gain_vec.size != self.nmodes:
                    gain_vec = self.xp.broadcast_to(gain_vec, (self.nmodes,)).astype(self.dtype)
            
            # gain_vec = self.xp.array([0.3])
            self.iir_filter_data.set_gain(gain_vec)
            # print(gain_vec.shape);exit()
            # self.iir_filter_data.set_gain([0.7])
        
        
            