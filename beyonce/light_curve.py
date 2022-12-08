# generic modules
import numpy as np
import matplotlib.pyplot as plt

# beyonce modules
import validate



class LightCurve:
    """
    This class is used to group light curve attributes (time, flux, error), 
    period for folding and some basic behaviour
    behaviour.
    """


    def __init__(self, 
            time: np.ndarray, 
            flux: np.ndarray, 
            error: np.ndarray = None
        ) -> None:
        """
        This method is the constructor for the light curve, which simply
        sets the parameters up.

        Parameters
        ----------
        time : np.ndarray (float 1-D)
            These are the times measured in the light curve [day].
        flux : np.ndarray (float 1-D)
            These are the normalised fluxes measured in the light curve [-].
        error : np.ndarray (float 1-D)
            These are the flux errors measured in the light curve [-]. Note
            that this may be None.
        """

        self.time = validate.array(time, "time", dtype="float64", 
            num_dimensions=1)
        self.flux = validate.array(flux, "flux", dtype="float64",
            lower_bound=0., num_dimensions=1)
        if error is None:
            self.error = np.zeros_like(time)
        else:
            self.error = validate.array(error, "error", dtype="float64",
                lower_bound=0., num_dimensions=1)

        self._set_data(time, flux, error)
        self.fold_period: float = None
        self.phase: np.ndarray = None


    def __str__(self) -> str:
        """
        This method is used to print all the information pertaining to the 
        light curve object.

        Returns
        -------
        print_light_curve_class : str
            This contains the string representation of the light curve class.
        """
        baseline = self.time[-1] - self.time[0]
        min_flux = np.nanmin(self.flux)
        max_flux = np.nanmax(self.flux)
        median_flux = np.nanmedian(self.flux)
        median_error = np.nanmedian(self.error)

        lines = []
        lines.append("\n======================================")
        lines.append("************ LIGHT CURVE *************")
        lines.append("======================================")
        lines.append("\nPhotometry")
        lines.append("==========\n")
        lines.append(f"Number of Points: {self.num_points}")
        lines.append(f"Base Line: {baseline:.2f}\n")
        lines.append(f"Flux (min): {min_flux:.4f}")
        lines.append(f"Flux (max): {max_flux:.4f}")
        lines.append(f"Flux (median): {median_flux:.4f}")
        lines.append(f"\nError (median): {median_error:.4f}")
        lines.append("")
        
        if self.fold_period is not None:
            lines.append(f"Fold Period: {self.fold_period:.2f}")
        
        lines.append("\n======================================")

        print_light_curve_class = "\n".join(lines)

        return print_light_curve_class


    def _set_data(self, 
            time: np.ndarray, 
            flux: np.ndarray, 
            error: np.ndarray = None
        ) -> None:
        """
        This method sets the data fields of the light curve (time, flux and 
        error)

        Parameters
        ----------
        time : np.ndarray (float 1-D)
            These are the times measured in the light curve [day].
        flux : np.ndarray (float 1-D)
            These are the normalised fluxes measured in the light curve [-].
        error : np.ndarray (float 1-D)
            These are the flux errors measured in the light curve [-]. Note
            that this may be None.
        """
        time = validate.array(time, "time", num_dimensions=1, dtype="float64")
        flux = validate.array(flux, "flux", num_dimensions=1, dtype="float64",
            lower_bound=0.)
        
        if error is not None:
            error = validate.array(error, "error", num_dimensions=1, 
                dtype="float64", lower_bound=0.)
        else:
            error = np.zeros_like(time)

        arrays_list = [time, flux, error]
        names_list = ["time", "flux", "error"]
        validate.same_shape_arrays(arrays_list, names_list)
        
        # sort by time
        sort_mask = np.argsort(time)
        self.time = time[sort_mask]
        self.flux = flux[sort_mask]
        self.error = error[sort_mask]
        self.num_points = len(time)


    def get_data(self, 
            folded: bool = False
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method is used to retrieve the data on light curve object. It
        retrieves either the time or phase light depending on the folded
        parameter.

        Parameters
        ----------
        folded : bool
            Whether to retrieve the folded or the original light curve 
            [default = False].

        Returns
        -------
        time or phase : np.ndarray (float 1-D)
            These are the times or phases measured in the light curve 
            [day, -].
        flux : np.ndarray (float 1-D)
            These are the normalised fluxes measured in the light curve [-].
        error : np.ndarray (float 1-D)
            These are the flux errors measured in the light curve [-]. Note
            that this may be None.
        """
        folded = validate.boolean(folded, "folded")
        
        if folded and self.phase is None:
            raise RuntimeError("to retrieve folded data the light curve must"
                " be folded using .fold() method")
            
        if folded:
            return self.phase, self.flux, self.error

        return self.time, self.flux, self.error


    def fold(self, period: float) -> None:
        """
        This method folds the light curve with the provided period. This is
        done such that the phase goes from -0.5 to 0.5.
        
        Parameters
        ----------
        period : float
            This is the period with which to fold the data [day].
        """
        self.fold_period = validate.number(period, "period", lower_bound=0)
        self.phase = (self.time % self.fold_period) / self.fold_period - 0.5
    
    def plot(self, 
            ax: plt.Axes = None, 
            folded: bool = False, 
            label: str = "",
            color: str = "r", 
            line_style: str = "", 
            marker: str = ".", 
            alpha: float = 1
        ) -> plt.Axes:
        """
        This method is used to plot the light curve.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes object that will contain the plot.
        folded : bool
            Whether to plot the folded or the original light curve 
            [default = False].
        label : string
            Label assigned to the light curve plotted [default = ""].
        color : string
            Matplotlib color to assign the light curve data [default = "r"].
        line_style : string
            Matplotlib line styles [default = "-"].
        marker : string
            Matplotlib marker [default = "."]
        alpha : float
            Determines the transparency of the data [default = 1].

        Returns
        -------
        ax : matplotlib.Axes
            Axes object that contains the plot.
        """
        if ax is None:
            ax = plt.gca()
        ax = validate.class_object(ax, "ax", plt.Axes, "Axes")

        folded = validate.boolean(folded, "folded")
        label = validate.string(label, "label")
        color = validate.string(color, "color")
        line_style = validate.string(line_style, "line_style")
        marker = validate.string(marker, "marker")
        alpha = validate.number(alpha, "alpha", lower_bound=0, upper_bound=1)
        
        time_or_phase, flux, error = self.get_data(folded)
        
        # plot the light curve
        ax.errorbar(time_or_phase, flux, yerr=error, fmt=marker, color=color,
            ls=line_style, lw=3, alpha=alpha, label=label)

        # set axes labels and limits
        if folded:
            ax.set_xlabel("Phase [-]")
        else:
            ax.set_xlabel("Time [day]")
        ax.set_ylabel("Normalised Flux [-]")

        return ax