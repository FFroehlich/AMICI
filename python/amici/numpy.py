import numpy as np

def fieldAsNumpy(fieldDimensions, field, data):
    """ Convert data object field to numpy array with dimensions according to specified field dimensions

    Arguments:
        fieldDimensions: dimension specifications dict({field: list([dim1, dim2, ...])})
        data: object with fields
        field: Name of field

    Returns:
        Field Data as numpy array with dimensions according to specified field dimensions

    Raises:

    """
    attr = getattr(data, field)
    if field in fieldDimensions.keys():
        if len(attr) == 0:
            return None
        else:
            return np.array(attr).reshape(fieldDimensions[field])
    else:
        return float(attr)


def rdataToNumPyArrays(rdata):
    """ Convenience wrapper ReturnData class (generated by swig)

    Arguments:
        rdata: ReturnData instance with simulation results

    Returns:
        ReturnData object with numpy array fields

    Raises:

    """
    npReturnData = {'ptr': rdata}
    fieldNames = ['t', 'x', 'x0', 'sx', 'sx0', 'y', 'sigmay', 'sy', 'ssigmay',
                  'z', 'rz', 'sigmaz', 'sz', 'srz', 'ssigmaz', 'sllh', 's2llh',
                  'J', 'xdot', 'status', 'llh', 'chi2', 'res', 'sres', 'FIM',
                  'wrms_steadystate', 't_steadystate',
                  'newton_numlinsteps', 'newton_numsteps',
                  'numsteps', 'numrhsevals', 'numerrtestfails',
                  'numnonlinsolvconvfails',
                  'order', 'numstepsB', 'numrhsevalsB', 'numerrtestfailsB',
                  'numnonlinsolvconvfailsB']

    for field in fieldNames:
        npReturnData[field] = getReturnDataFieldAsNumPyArray(rdata, field)

    return npReturnData


def edataToNumPyArrays(edata):
    """ Convenience wrapper ExpData class (generated by swig)

    Arguments:
        edata: ExpData instance with experimental data

    Returns:
        ExpData object with numpy array fields

    Raises:

    """
    npExpData = {'ptr': edata}

    fieldNames = ['observedData', 'observedDataStdDev', 'observedEvents',
                  'observedEventsStdDev', 'fixedParameters',
                  'fixedParametersPreequilibration']

    edata.observedData = edata.getObservedData()
    edata.observedDataStdDev = edata.getObservedDataStdDev()
    edata.observedEvents = edata.getObservedEvents()
    edata.observedEventsStdDev = edata.getObservedEventsStdDev()

    for field in fieldNames:
        npExpData[field] = getExpDataFieldAsNumPyArray(edata, field)

    return npExpData


def getReturnDataFieldAsNumPyArray(rdata, field):
    """ Convert ReturnData field to numpy array with dimensions according to model dimensions in rdata

    Arguments:
        rdata: ReturnData instance with simulation results
        field: Name of field

    Returns:
        Field Data as numpy array with dimensions according to model dimensions in rdata

    Raises:

    """

    fieldDimensions = {'ts': [rdata.nt],
                       'x': [rdata.nt, rdata.nx],
                       'x0': [rdata.nx],
                       'sx': [rdata.nt, rdata.nplist, rdata.nx],
                       'sx0': [rdata.nplist, rdata.nx],

                       # observables
                       'y': [rdata.nt, rdata.ny],
                       'sigmay': [rdata.nt, rdata.ny],
                       'sy': [rdata.nt, rdata.nplist, rdata.ny],
                       'ssigmay': [rdata.nt, rdata.nplist, rdata.ny],

                       # event observables
                       'z': [rdata.nmaxevent, rdata.nz],
                       'rz': [rdata.nmaxevent, rdata.nz],
                       'sigmaz': [rdata.nmaxevent, rdata.nz],
                       'sz': [rdata.nmaxevent, rdata.nplist, rdata.nz],
                       'srz': [rdata.nmaxevent, rdata.nplist, rdata.nz],
                       'ssigmaz': [rdata.nmaxevent, rdata.nplist, rdata.nz],

                       # objective function
                       'sllh': [rdata.nplist],
                       's2llh': [rdata.np, rdata.nplist],

                       'res': [rdata.nt * rdata.nytrue],
                       'sres': [rdata.nt * rdata.nytrue, rdata.nplist],
                       'FIM': [rdata.nplist, rdata.nplist],

                       # diagnosis
                       'J': [rdata.nx_solver, rdata.nx_solver],
                       'xdot': [rdata.nx_solver],
                       'newton_numlinsteps': [rdata.newton_maxsteps, 2],
                       'newton_numsteps': [1, 2],
                       'numsteps': [rdata.nt],
                       'numrhsevals': [rdata.nt],
                       'numerrtestfails': [rdata.nt],
                       'numnonlinsolvconvfails': [rdata.nt],
                       'order': [rdata.nt],
                       'numstepsB': [rdata.nt],
                       'numrhsevalsB': [rdata.nt],
                       'numerrtestfailsB': [rdata.nt],
                       'numnonlinsolvconvfailsB': [rdata.nt],
                       }
    if field == 't':
        field = 'ts'

    return fieldAsNumpy(fieldDimensions, field, rdata)


def getExpDataFieldAsNumPyArray(edata, field):
    """ Convert ExpData field to numpy array with dimensions according to model dimensions in edata

    Arguments:
        edata: ExpData instance with experimental data
        field: Name of field

    Returns:
        Field Data as numpy array with dimensions according to model dimensions in edata

    Raises:

    """

    fieldDimensions = {  # observables
        'observedData': [edata.nt(), edata.nytrue()],
        'observedDataStdDev': [edata.nt(), edata.nytrue()],

        # event observables
        'observedEvents': [edata.nmaxevent(), edata.nztrue()],
        'observedEventsStdDev': [edata.nmaxevent(), edata.nztrue()],

        # fixed parameters
        'fixedParameters': [len(edata.fixedParameters)],
        'fixedParametersPreequilibration': [
            len(edata.fixedParametersPreequilibration)],
    }

    return fieldAsNumpy(fieldDimensions, field, edata)