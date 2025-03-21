import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def BHCT_KT_formula(Gamma, T_fb):
    """
    Approximate the bottom-hole circulating temperature from an
    empirical Kutasov-Targhi equation.

    This formula computes the BHCT value using a linear combination of
    Gamma (the static temperature gradient) and T_fb (the bottom-hole
    static undisturbed temperature) along with fixed coefficients.

    Reference:
        Kutasov, I.M. and Targhi, A.K.,
        “Better Deep-Hole BHCT Estimations Possible”,
        OG J., May 25, 1987, pp. 71-73.

    Parameters:
        Gamma (float): The static temperature gradient, in [degF/ft].
        T_fb (float): The bottom-hole static (undisturbed) temperature, in degrees Fahrenheit [degF].

    Returns:
        float: The calculated BHCT value in degrees Fahrenheit [degF].

    Formula:
        BHCT = d1 + d2 * Gamma + (d3 - d4 * Gamma) * T_fb

    Constants:
        d1 = -102.1      # Offset in degrees Fahrenheit [degF].
        d2 = 3354        # Coefficient for Gamma in [ft].
        d3 = 1.342       # Constant multiplier.
        d4 = 22.28       # Coefficient for Gamma in [ft/degF].
    """
    d1 = -102.1  # [degF]
    d2 = 3354    # [ft]
    d3 = 1.342
    d4 = 22.28   # [ft/degF]

    # Check if T_fb is within the valid range
    if T_fb < 166 or T_fb > 414:
        warnings.warn("Method is a valid approximation for 166 degF <= T_fb <= 414 degF.", UserWarning)

    # Check if Gamma is within the valid range
    if Gamma < 0.83/100 or Gamma > 2.44/100:
        warnings.warn("Method is a valid approximation for 0.83 degF/100ft <= Gamma <= 2.44 degF/100ft.", UserWarning)

    return d1 + d2 * Gamma + (d3 - d4 * Gamma) * T_fb

def T_fb(H, Gamma, T0, Hw=0):
    """
    Calculate the bottom-hole static (undisturbed) temperature (T_fb)
    using the temperature gradient and depth.

    This formula estimates the temperature at a specified depth based on
    the surface formation temperature, temperature gradient, and a reference
    water depth.

    Parameters:
        H (float): The true vertical depth of interest in feet [ft].
        Gamma (float): The static temperature gradient in degrees Fahrenheit per foot [degF/ft].
        T0 (float): The surface formation temperature in degrees Fahrenheit [degF].
        Hw (float, optional): The depth of water in feet [ft] (default is 50 ft).

    Returns:
        float: The calculated bottom-hole static temperature (T_fb) in degrees Fahrenheit [degF].

    Formula:
        T_fb = T0 + Gamma * (H - Hw)

    Notes:
        - Ensure that the depth `H` is greater than the water depth `Hw` for meaningful results.
        - Adjust `Hw` as needed based on actual well data.
    """
    return T0 + Gamma * (H - Hw)

def API_temperature_correlation(H, Gamma):
    """
    Estimate the bottom-hole static temperature at a given depth and temperature gradient
    using the API temperature correlation method.

    This function interpolates a pre-defined grid of temperature values based on depth and
    temperature gradient to provide an estimated temperature for the specified values.

    The grid contains data for various depths (H) and static temperature gradients (Gamma),
    with interpolated values for corresponding temperatures at those conditions.

    Parameters:
        H (float): The true vertical depth of the well in feet [ft].
        Gamma (float): The static temperature gradient in degrees Fahrenheit per foot [degF/ft].

    Returns:
        float: The interpolated bottom-hole static temperature in degrees Fahrenheit [degF].

    Notes:
        - This method assumes that depth `H` and temperature gradient `Gamma` fall within the pre-defined grid values.
        - The `RegularGridInterpolator` is used to perform linear interpolation.

    Example:
        temperature = API_temperature_correlation(12000, 0.013)
    """
    Gammas = np.array([0.9, 1.1, 1.3, 1.5, 1.7, 1.9])/100
    Hs = np.array([8000, 10000, 12000, 14000, 16000, 18000, 20000])
    temperatures = np.array([
        [118, 129, 140, 151, 162, 173],
        [132, 147, 161, 175, 189, 204],
        [148, 165, 183, 201, 219, 236],
        [164, 185, 207, 228, 250, 271],
        [182, 207, 233, 258, 284, 309],
        [201, 231, 261, 291, 321, 350],
        [222, 236, 291, 326, 360, 395]
    ])

    interp_func = RegularGridInterpolator((Hs, Gammas), temperatures,  method='linear', bounds_error=False, fill_value=None)
    return interp_func([[H, Gamma]])[0]

class API_EW:
    """
    A class that estimates and calculates temperature-related parameters
    for wells based on the API temperature correlation, Kutasov-Targhi formula,
    and other related methods.

    The class provides functionality to estimate the static temperature gradient (Gamma)
    and various temperature parameters like Bottom-Hole Static Temperature (BHST) and
    Bottom-Hole Circulating Temperature (BHCT), using the provided depth (H), temperature
    gradient (Gamma), and other well data (e.g., surface temperature, water depth, seabed temperature).

    Attributes:
        H (float): The true vertical depth of the well in feet [ft].
        Gamma (float, optional): The static temperature gradient in degrees Fahrenheit per foot [degF/ft].
        T0 (float, optional): The surface formation temperature in degrees Fahrenheit [degF].
        Hw (float, optional): The depth of water in feet [ft].
        Tw (float, optional): The temperature at the seabed in degrees Fahrenheit [degF].
    """

    def __init__(self, H, Gamma=None, T0=None, Hw=None, Tw=None):
        """
        Initializes an instance of the API_EW class.

        Parameters:
            H (float): The true vertical depth of the well in feet [ft].
            Gamma (float, optional): The static temperature gradient in degrees Fahrenheit per foot [degF/ft].
            T0 (float, optional): The surface formation temperature in degrees Fahrenheit [degF].
            Hw (float, optional): The depth of water in feet [ft].
            Tw (float, optional): The temperature at the seabed in degrees Fahrenheit [degF].
        """
        self.H = H
        if Gamma is None:
            self._gamma_set = False
        else:
            self.Gamma = Gamma
            self._gamma_set = True
        self.Hw = Hw
        self.T0 = T0
        self.Tw = Tw

    def __getattr__(self, name):
        """
        Custom behavior for getting the 'Gamma' attribute.

        If 'Gamma' is accessed but not set, raises an AttributeError.

        Parameters:
            name (str): The attribute name to access.

        Returns:
            The value of the attribute or raises an AttributeError.
        """
        if name == 'Gamma':
            if not self._gamma_set:
                raise AttributeError(f"Gamma has to be either set or estimated before.")
        return super().__getattribute__(name)

    def estimate_Gamma_from_BHCT(self, BHCT):
        """
        Estimates the static temperature gradient (Gamma) from the given
        Bottom-Hole Circulating Temperature (BHCT) using the Kutasov-Targhi formula.

        Parameters:
            BHCT (float): The Bottom-Hole Circulating Temperature in degrees Fahrenheit [degF].

        Returns:
            float: The estimated static temperature gradient (Gamma) in degrees Fahrenheit per foot [degF/ft].
        """
        if self._gamma_set:
           warnings.warn("Gamma set before. This estimation changed its value.", UserWarning)
        if self.Hw is None:
            self.Hw = 0
            warnings.warn("Hw set to 0 as no value was given.", UserWarning)
        if self.Hw > 0:
            if self.Tw is None:
                self.Tw = 40
                warnings.warn("Tw set to 40 as no value was given.", UserWarning)
            T_ref = self.Tw
        else:
            if self.T0 is None:
                self.T0 = 80
                warnings.warn("T0 set to 80 as no value was given.", UserWarning)
            T_ref = self.T0

        d1, d2, d3, d4 = (-102.1, 3354, 1.342, 22.28) # parameters from KT Formula

        B1 = (d4 * T_ref - d2 - d3 * (self.H - self.Hw)) / (d4 * (self.H - self.Hw))
        B0 = (BHCT - d1 - d3 * T_ref) / (d4 * (self.H - self.Hw))
        Gamma = -B1 / 2 - (B1**2 / 4 - B0) ** 0.5
        self.Gamma = Gamma
        self._gamma_set = True
        return Gamma

    @staticmethod
    def H_equiv(H, Gamma, H0=None, T0=None):
        """
        Calculates the equivalent depth (H_equiv) based on the static temperature gradient (Gamma) and surface temperature.

        Parameters:
            H (float): The true vertical depth of the well in feet [ft].
            Gamma (float): The static temperature gradient in degrees Fahrenheit per foot [degF/ft].
            H0 (float, optional): The reference depth, default is 0.
            T0 (float, optional): The surface temperature in degrees Fahrenheit [degF], default is 80.

        Returns:
            float: The equivalent depth in feet [ft].
        """
        if H0 is None:
            H0 = 0
        if T0 is None:
            T0 = 80
        if H < H0:
            return 0
        return H - H0 + (T0 - 80) / Gamma

    def get_H_equiv(self):
        """
        Returns the equivalent depth (H_equiv) based on the current object's attributes.

        Returns:
            float: The equivalent depth in feet [ft].
        """
        H = self.H
        Gamma = self.Gamma
        H0 = 0 if self.Hw is None else self.Hw
        if H0 > 0:
            T0 = 40 if self.Tw is None else self.Tw
        else:
            T0 = 80 if self.T0 is None else self.T0
        return self.H_equiv(H, Gamma, H0, T0)

    def get_BHST(self):
        """
        Returns the estimated Bottom-Hole Static Temperature (BHST) for the well
        based on the current attributes, using the temperature gradient and depth.

        Returns:
            float: The Bottom-Hole Static Temperature in degrees Fahrenheit [degF].
        """
        H = self.H
        Gamma = self.Gamma
        T0 = 80 if self.T0 is None else self.T0
        Hw = 0 if self.Hw is None else self.Hw
        Tw = 40 if self.Tw is None else self.Tw

        if Hw > 0:
            return T_fb(H, Gamma, Tw, Hw)
        else:
            return T_fb(H, Gamma, T0)

    def get_BHCT_KT(self):
        """
        Returns the estimated Bottom-Hole Circulating Temperature (BHCT) using the
        Kutasov-Targhi formula based on the static temperature gradient and Bottom-Hole Static Temperature.

        Returns:
            float: The estimated Bottom-Hole Circulating Temperature in degrees Fahrenheit [degF].
        """
        Gamma = self.Gamma
        T_bf = self.get_BHST()
        return BHCT_KT_formula(Gamma, T_bf)

    def get_BHCT(self):
        """
        Returns the estimated Bottom-Hole Circulating Temperature (BHCT) based on
        the equivalent depth and static temperature gradient using the API temperature correlation.

        Returns:
            float: The estimated Bottom-Hole Circulating Temperature in degrees Fahrenheit [degF].
        """
        Gamma = self.Gamma
        H_equiv = self.get_H_equiv()
        return API_temperature_correlation(H_equiv, Gamma)
