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
                 noise_variance: float = 100,
                 ar_order: int = 20,
                 rho: float = 1e3,
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
        self.ar_order = int(ar_order)
        self.rho = float(rho)

        self.noise_variance = noise_variance

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

        self.A, self.B, self.C, self.D, self.A_T, self.B_T, self.C_T, self.D_T, self.H_cl = self._generate_state_space() #generate state space matrices for the closed loop system

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
        orig_gain = float(cpuArray(self.iir_filter_data.gain[0]))
        # orig_gain = g
        num_normalized = self.xp.asarray(num, dtype=float) / orig_gain
        num = num_normalized * g  # Scale numerator by symbolic gain g for optimization
        # Determine integer delays (in samples). Use rounding to allow non-integer attributes.
        total_delay = int(round(self.delay))
        sensor_delay = total_delay // 2
        actuator_delay = total_delay - sensor_delay

        # Calculate regulator transfer function R(z) with symbolic gain included.
        num_val = self.xp.polyval(num[::-1], z)
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
        T_cl = open_loop / denominator  # complementary sensitivity (from reference to output)

        
        # Extract and print the coefficients of z in the numerator and denominator of H

        # sensitivity realization
        num_expr, den_expr = sp.fraction(sp.together(H_cl))
        num_poly = sp.Poly(num_expr, z)
        den_poly = sp.Poly(den_expr, z)

        lead = den_poly.LC()
        den_poly = sp.Poly(den_poly.as_expr() / lead, z)
        num_poly = sp.Poly(num_poly.as_expr() / lead, z)

        n = den_poly.degree()
        den_coeffs = den_poly.all_coeffs()

        A = sp.zeros(n)
        for i in range(n - 1):
            A[i, i + 1] = 1
        A[n - 1, :] = sp.Matrix([-den_coeffs[-(j + 1)] for j in range(n)]).T

        B = sp.zeros(n, 1)
        B[n - 1] = 1

        num_coeffs = num_poly.all_coeffs()
        if len(num_coeffs) < n:
            num_coeffs = [0] * (n - len(num_coeffs)) + num_coeffs

        # PATCH: do NOT reverse
        C = sp.Matrix(num_coeffs).T
        D = sp.sympify(0)

        # complementary sensitivity realization
        num_T, den_T = sp.fraction(sp.together(T_cl))
        num_T_poly = sp.Poly(num_T, z)
        den_T_poly = sp.Poly(den_T, z)

        lead_T = den_T_poly.LC()
        den_T_poly = sp.Poly(den_T_poly.as_expr() / lead_T, z)
        num_T_poly = sp.Poly(num_T_poly.as_expr() / lead_T, z)

        # PATCH: extract feedthrough like offline
        quotient_T, remainder_T = sp.div(num_T_poly.as_expr(), den_T_poly.as_expr(), z)
        D_T = sp.sympify(quotient_T)
        num_T_poly_proper = sp.Poly(remainder_T, z)

        n_T = den_T_poly.degree()
        den_T_coeffs = den_T_poly.all_coeffs()

        A_T = sp.zeros(n_T)
        for i in range(n_T - 1):
            A_T[i, i + 1] = 1
        A_T[n_T - 1, :] = sp.Matrix([-den_T_coeffs[-(j + 1)] for j in range(n_T)]).T

        B_T = sp.zeros(n_T, 1)
        B_T[n_T - 1] = 1

        num_T_coeffs = num_T_poly_proper.all_coeffs()
        if len(num_T_coeffs) < n_T:
            num_T_coeffs = [0] * (n_T - len(num_T_coeffs)) + num_T_coeffs

        # PATCH: do NOT reverse
        C_T = sp.Matrix(num_T_coeffs).T

        print("Complementary Sensitivity numerator polynomial:", num_T_poly)
        print("Complementary Sensitivity denominator polynomial:", den_T_poly)

        self.x = self.xp.zeros((n, self.n_realizations), dtype=self.dtype)  # State vector placeholder
        # A, B, C, D = iir_filter_data.to_state_space()
        return A, B, C, D, A_T, B_T, C_T, D_T, H_cl
    
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
        # Match offline SISO behavior while handling both 1D and multi-component exogenous inputs.
        data = self.xp.asarray([
            self.xp.asarray(d).ravel()[1] if self.xp.asarray(d).ravel().size > 1 else self.xp.asarray(d).ravel()[0]
            for d in self.data_history
        ][-data_to_consider:])
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

        turb_pred = self._predict_turbulent_signal(data, order=self.ar_order) # last opt_dt values to learn
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

    


    def _exo_id_autoreg(self, data,order):
        """
        Identification of the exosystem with an autoregressive model of a given order
        """
        model = AutoReg(data, lags=order, exog=np.random.normal(0, 1, size=(len(data), 1)))
        model_fit = model.fit()
        ar_params = model_fit.params[1:1+order]  # exclude intercept
        exog_params = model_fit.params[1+order:]  # exogenous parameters if needed
        beta_exog = float(exog_params[0]) if exog_params.size > 0 else 0.0
        noise_std = self.xp.std(model_fit.resid)
        return model_fit, noise_std, beta_exog, ar_params
    

    def _predict_turbulent_signal(self, data, order):
        """
        predict the turbulent signal in the predictive optimization horizon (opt_dt)
        """
        model_fit, noise_std, beta_exog, ar_params = self._exo_id_autoreg(data, order=order)
        # Get the noise standard deviation from the model
        # Generate multiple realizations of the future signal

        y  = self.xp.zeros((self.n_realizations, self.prediction_horizon), dtype=self.dtype)
        y[:,:order] = self.xp.repeat(data[-order:].reshape(1, -1), self.n_realizations, axis=0).astype(self.dtype)  # Initialize with the last 'order' values from the data
        exog_white_noise = self.xp.random.normal(0, 1, size=(self.n_realizations, self.prediction_horizon)).astype(self.dtype)
        ar_innovation = self.xp.random.normal(0, noise_std, size=(self.n_realizations, self.prediction_horizon)).astype(self.dtype)
        for i in range(self.n_realizations):
            for j in range(order, self.prediction_horizon):
                #constant term
                y[i, j] = model_fit.params[0]
                for k in range(1, order+1):
                    y[i, j] += ar_params[k-1] * y[i, j-k]
                y[i, j] += beta_exog * exog_white_noise[i, j] + ar_innovation[i, j]

        if self.plot_debug:
            plt.figure(figsize=(10, 4 + 1 * self.n_realizations))
            y_np = np.asarray(cpuArray(y))
            for i in range(self.n_realizations):
                plt.plot(y_np[i], alpha=0.55, label=f"Realization {i+1}")
            plt.plot(np.mean(y_np, axis=0), color="k", linewidth=2.0, label="Mean prediction")
            plt.xlabel("Prediction step")
            plt.ylabel("Predicted signal y")
            plt.title(f"Predicted turbulent signal over horizon (order={order})")
            plt.grid(True)
            plt.legend(loc="upper right", fontsize="small")
            plt.tight_layout()
            plt.show()

            past_data = np.asarray(cpuArray(data)).ravel()
            if past_data.size >= 2 and y_np.shape[1] >= 2:
                nperseg = min(1024, past_data.size, y_np.shape[1])
                fs = 1.0 / float(self.time_step)

                freq_past, psd_past = signal.welch(past_data, fs=fs, nperseg=nperseg)
                psd_pred_accum = None
                for i in range(self.n_realizations):
                    freq_pred, psd_pred = signal.welch(y_np[i], fs=fs, nperseg=nperseg)
                    if psd_pred_accum is None:
                        psd_pred_accum = psd_pred
                    else:
                        psd_pred_accum += psd_pred
                psd_pred_mean = psd_pred_accum / self.n_realizations

                plt.figure(figsize=(10, 5))
                plt.loglog(freq_past, psd_past, label="Past data PSD", linewidth=2.0)
                plt.loglog(freq_pred, psd_pred_mean, label="Predicted PSD (mean)", linewidth=2.0)
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("PSD")
                plt.title(f"Predicted vs past-data PSD (order={order})")
                plt.grid(True, which="both", alpha=0.3)
                plt.legend(loc="upper right", fontsize="small")
                plt.tight_layout()
                plt.show()
            
        return y

    def _optimize_g(self, predicted_signal, max_gain):
        """
        Optimize gain based on predicted turbulent signal.
        """

        n_points = predicted_signal.shape[1]
        noise_seq = self.xp.random.randn(self.n_realizations, n_points).astype(self.dtype)

        min_gain = 0.1 if float(max_gain) > 0.1 else 0.0

        res = optimize.minimize_scalar(
            lambda gg: self._cost_function(
                gg,
                predicted_signal,
                noise_sequence_unit=noise_seq,
            ),
            bounds=(min_gain, float(max_gain)),
            method='bounded',
            options={"maxiter": int(1e5), "xatol": 1e-6},
        )
        return res.x
    
    def _cost_function(self, gain, predicted_signal, noise_sequence_unit=None):
        if hasattr(gain, '__len__'):
            gain = float(gain[0])
        else:
            gain = float(gain)

        g = sp.symbols('g', real=True, positive=True)

        A = self.xp.array(self.A.subs(g, gain), dtype=self.dtype)
        B = self.xp.array(self.B.subs(g, gain), dtype=self.dtype)
        C = self.xp.array(self.C.subs(g, gain), dtype=self.dtype)

        A_T = self.xp.array(self.A_T.subs(g, gain), dtype=self.dtype)
        B_T = self.xp.array(self.B_T.subs(g, gain), dtype=self.dtype)
        C_T = self.xp.array(self.C_T.subs(g, gain), dtype=self.dtype)
        D_T = self.dtype(self.D_T.subs(g, gain))

        n_points = predicted_signal.shape[1]

        if noise_sequence_unit is None:
            noise_sequence_unit = self.xp.random.randn(self.n_realizations, n_points).astype(self.dtype)

        x = self.xp.zeros((A.shape[0], self.n_realizations), dtype=self.dtype)
        x_n = self.xp.zeros((A_T.shape[0], self.n_realizations), dtype=self.dtype)

        cost_tracking = self.dtype(0.0)
        cost_noise = self.dtype(0.0)
        y = self.xp.zeros((self.n_realizations, n_points), dtype=self.dtype)    
        y_n = self.xp.zeros((self.n_realizations, n_points), dtype=self.dtype)

        for k in range(n_points):
            for j in range(self.n_realizations):
                u = predicted_signal[j, k]
                y[j, k] = (C @ x[:, j:j+1])[0, 0]
                # cost_tracking += y[j, k]**2
                x[:, j:j+1] = A @ x[:, j:j+1] + B * u

                noise_input = noise_sequence_unit[j, k] * self.noise_variance
                y_n[j, k] = (C_T @ x_n[:, j:j+1])[0, 0] + D_T * noise_input
                # cost_noise += y_n[j, k]**2
                x_n[:, j:j+1] = A_T @ x_n[:, j:j+1] + B_T * noise_input

        if self.plot_debug:
            plt.figure(figsize=(10, 4))
            plt.plot(y_n[0, :], label="System output (signal + noise)")
            plt.ylim([-500, 500])
            plt.xlabel("Time step")
            plt.ylabel("Output")
            plt.title(f"System response to predicted signal and noise (gain={gain:.4f})")
            plt.grid(True)
            plt.legend(loc="upper right", fontsize="small")
            plt.tight_layout()
            plt.show()
        
        #define the cost 
        y_tot = y + y_n
        steady_state_interval = slice(n_points//2, n_points)  # Consider only the second half of the prediction horizon for cost evaluation
        cost_tracking = self.xp.mean(y_tot[:,steady_state_interval]**2)

        rho = self.dtype(self.rho)
        return cost_tracking + rho * gain**2

        
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
        
        
            