import pandas as pd

class AtmosData:
    """
    AtmosData class

    Loads atmospheric and oceanographic data from the specified data file
    and stores the relevant variables as attributes.

    Initialized via `AtmosData(<dataset>)`, where `dataset` can either be
    an xarray dataset or a string specifying the data file location.    
    """

    def __init__(self, dataset):
        if type(dataset) == pd.core.frame.DataFrame:
            self.ds = dataset
        elif type(dataset) == str:
            self.ds = xr.load_dataset(dataset, decode_timedelta=False)
        
        self.time = self.ds["time"].to_numpy()
        self.T_skin = self.ds["skinsst"].to_numpy()
        self.dsst_err = self.ds["dsst_err"].to_numpy()
        self.T_f = self.ds["ftemp"].to_numpy()
        self.u = self.ds["wind"].to_numpy()
        self.T_a_rel = self.ds["atemp_rel"].to_numpy()
        self.Q_sw = self.ds["swrad"].to_numpy()
        self.q_v = self.ds["humid"].to_numpy()