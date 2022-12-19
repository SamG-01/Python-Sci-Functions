from numpy import nan
from uncertainties.unumpy import nominal_values, std_devs
from scipy.odr import RealData, Model, ODR

def odr_fit(model, xdata, ydata, beta0):
    """
    Fits a set of y vs x data to a model function using scipy.odr.
    
    Arguments:
    model {function} -- the function to fit to. Should be of the form func(p, x), where p is a vector of parameters.
    xdata {array-like} -- a 1D array of ufloat objects for the input variable.
    ydata {array-like} -- a 1D array of ufloat objects for the response variable.
    beta0 {iterable} -- an iterable containing the initial guess for the fit parameters.
    """

    # Splits values and uncertainties
    x, y, dx, dy = nominal_values(xdata), nominal_values(ydata), std_devs(xdata), std_devs(ydata)

    # Replaces points with zero uncertainty with nan to avoid errors
    dx[dx == 0] = nan
    dy[dy == 0] = nan

    # Puts data and model into SciPy ODR objects
    fitdata = RealData(x, y, dx, dy)
    fitmodel = Model(model)
    
    # Sets up and runs the regression
    odr = ODR(fitdata, fitmodel, beta0=beta0)
    out = odr.run()

    return out
