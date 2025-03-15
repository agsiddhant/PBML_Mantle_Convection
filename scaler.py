import numpy as np


def scale_var(x, raq, fkt, fkp, var):
    if var == "uprev":
        scaler = (
            np.exp(
                (raq / 10) * 1.80167667
                + np.log(fkt) * 0.4330392
                + np.log(fkp) * -0.46052953
            )
            * 5
        )
        x /= scaler

    elif var == "vprev":
        scaler = (
            np.exp(
                (raq / 10) * 1.80167667
                + np.log(fkt) * 0.4330392
                + np.log(fkp) * -0.46052953
            )
            * 5
        )
        x /= scaler

    elif var == "pprev":
        x = x

    elif var == "Vprev":
        x = x  # (np.log(x) + 16.)/16.

    elif "Tprev" in var:
        x = x

    return x


def unscale_var(x, raq, fkt, fkp, var):
    if var == "uprev":
        scaler = (
            np.exp(
                (raq / 10) * 1.80167667
                + np.log(fkt) * 0.4330392
                + np.log(fkp) * -0.46052953
            )
            * 5
        )
        x *= scaler

    elif var == "vprev":
        scaler = (
            np.exp(
                (raq / 10) * 1.80167667
                + np.log(fkt) * 0.4330392
                + np.log(fkp) * -0.46052953
            )
            * 5
        )
        x *= scaler

    elif var == "pprev":
        x = x

    elif var == "Vprev":
        x = x  # (np.exp(x*16.)-16.)

    elif "Tprev" in var:
        x = x

    return x
