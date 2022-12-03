import numpy as np
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values, std_devs
from scipy.odr import RealData, Model, ODR
from scipy.stats import linregress

def linWithIntercept(p,x):
    a,b=p
    return a*x+b
def linWithoutIntercept(p,x):
    a=p
    return a*x

def linear_fit(xdata, ydata, intercept=True, odr_print=False):
    """Does a linear fit y=a*x+b for xdata, ydata (arrays of ufloats), or y=a*x if intercept is false."""
    x, y, dx, dy = nominal_values(xdata), nominal_values(ydata), std_devs(xdata), std_devs(ydata)
    f = linWithIntercept if intercept else linWithoutIntercept
    lin_regress = linregress(x, y)

    dx[dx == 0] = np.nan
    dy[dy == 0] = np.nan

    fitdata = RealData(x, y, dx, dy)

    guess = lin_regress[:2] if intercept else lin_regress[:1]

    line_model = Model(f)
    odr = ODR(fitdata, line_model, beta0=guess)
    out = odr.run()
    if odr_print:
        out.pprint()

    if intercept:
        a,b = out.beta
        da, db = out.sd_beta
        return ufloat(a, da), ufloat(b, db), out.res_var
    else:
        a = out.beta[0]
        da = out.sd_beta[0]
        return ufloat(a, da), out.res_var
