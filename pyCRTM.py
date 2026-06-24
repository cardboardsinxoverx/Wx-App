#!/usr/bin/env python3
import configparser
import logging
import os, sys, h5py, netCDF4, traceback
# --- CRITICAL: Add the directory containing pyCRTM.py to the search path ---
# 1. Define the directory path
PYCRTM_DIR = '/home/desoxyn/frostbyte/frostbyte_project/state maps'

# 2. Add it to the system path
if PYCRTM_DIR not in sys.path:
    sys.path.insert(0, PYCRTM_DIR)
    
# 3. Now you can import the file by its name (without .py)
try:
    import pyCRTM
    logging.info("Successfully imported pyCRTM.")
except ImportError as e:
    logging.error(f"Failed to import pyCRTM: {e}. Check if the file exists at {PYCRTM_DIR}/pyCRTM.py")
import numpy as np
thisDir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,thisDir)
from crtm_io import readSpcCoeff, readSpcCoeffNc, findLib, findLibDyld, setLD_LIBRARY_PATH,  setDYLD_LIBRARY_PATH

# sharedLibPath has a shared object, append to LD_LIBRARY_PATH

try:
    from pycrtm_ import pycrtm_ as p
except Exception as e1:
    sharedLibPath = findLib(thisDir)
    if(len(sharedLibPath) > 0):
        setLD_LIBRARY_PATH(sharedLibPath)
        from pycrtm_ import pycrtm_ as p
    elif(sys.platform=='darwin'):
        dyldPath = findLibDyld(thisDir)
        setDYLD_LIBRARY_PATH(dyldPath)
    else:
        print(traceback.format_exc())        
#print(sharedLibPath)
#print(os.environ.get('LD_LIBRARY_PATH'))
# load f2py pycrtm
pycrtm = p.pycrtm
from collections import namedtuple
# Absorber IDs taken from CRTM.
gases = {}
gases['Q']     = 1  # H2O for anyone not NWP focused ;)
gases['CO2']   = 2
gases['O3']    = 3
gases['N2O']   = 4
gases['CO']    = 5
gases['CH4']   = 6
gases['O2']    = 7
gases['NO']    = 8
gases['SO2']   = 9
gases['NO2']   = 10
gases['NH3']   = 11
gases['HNO3']  = 12
gases['OH']    = 13
gases['HF']    = 14
gases['HCl']   = 15
gases['HBr']   = 16
gases['HI']    = 17
gases['ClO']   = 18
gases['OCS']   = 19
gases['H2CO']  = 20
gases['HOCl']  = 21
gases['N2']    = 22
gases['HCN']   = 23
gases['CH3l']  = 24
gases['H2O2']  = 25
gases['C2H2']  = 26
gases['C2H6']  = 27
gases['PH3']   = 28
gases['COF2']  = 29
gases['SF6']   = 30
gases['H2S']   = 31
gases['HCOOH'] = 32

WATER_CLOUD = 1
ICE_CLOUD = 2
RAIN_CLOUD = 3
SNOW_CLOUD = 4
GRAUPEL_CLOUD = 5
HAIL_CLOUD = 6
RT_ADA = 56
RT_SOI = 168
RT_AMOM =  56
RT_VMOM =  62  


def profilesCreate( nProfiles, nLevels, nAerosols=1, nClouds=1, additionalGases=[] ):
    keys  = [ 'P', 'T', 'Q', 'O3']
    for g in additionalGases:
        if g in list(gases.keys()) and g not in keys:
            keys.append(g)
        elif g == 'H2O' or g.lower() == 'water' or g=='ozone':
            print("You worry too much, of course we have {}! Water and Ozone are always turned on.".format(g))
        else: 
            print("Warning! I don't know this gas: {}! I can't add it to the simulation!".format(g))
            print("You could pick one of these instead:")
            for gg in list(gases.keys()): print(gg)

    p = {}
    for k in list(keys):
        p[k] = np.nan*np.ones([nProfiles,nLevels])

    p['Pi'] = np.nan*np.ones([nProfiles, nLevels+1])
    # satzen, sataz, sunzen, sunaz, scanangle
    p['Angles'] = np.nan*np.ones([nProfiles, 5])
    # Salinity (PSU)
    p['Salinity'] = np.nan*np.zeros([nProfiles])
    # surftype, water type
    p['SurfType'] = np.nan*np.zeros([nProfiles,2])
    # latitude, longitude, elevation 
    p['SurfGeom'] = np.zeros([nProfiles,3])
    # yy, mm, dd, hh, mm, ss
    p['DateTimes'] = np.zeros([nProfiles,6], dtype=int) 
    p['DateTimes'][:,0] = 2001
    p['DateTimes'][:,1] = 1
    p['DateTimes'][:,2] = 1
    # concentration, effective radius
    if(nAerosols>0):
        p['aerosols'] = np.nan*np.ones([nProfiles, nLevels, nAerosols, 2])
        p['aerosolType'] =-1 *np.ones([nProfiles,nAerosols], dtype =int)
    
    # concentration, effective radius
    if(nClouds>0):
        p['clouds'] =  np.nan*np.ones([nProfiles, nLevels,  nClouds, 2])
        p['cloudType'] = -1 *np.ones([nProfiles,nClouds], dtype =int)
        p['cloudFraction'] = np.zeros([nProfiles,nLevels])

    p['LAI'] = np.zeros([nProfiles])
    # surface 
    p['surfaceTemperatures'] = np.zeros([nProfiles,4])
    p['surfaceFractions'] = np.zeros([nProfiles,4])
    # land, soil, veg, water, snow, ice
    p['surfaceTypes'] = np.zeros([nProfiles,6], dtype=int)
    p['climatology'] = 6*np.ones([nProfiles], dtype=int)  # use usstd as default climatology for unspecified layers to 0.005 mbar (crtm will fill in the gaps if the user doesn't)

    p['windSpeed10m'] = np.zeros([nProfiles])
    p['windDirection10m'] = np.zeros([nProfiles])

    profiles = namedtuple("Profiles", p.keys())(*p.values())
    return profiles

class pyCRTM:
    def __init__(self):
        thisDir = os.path.split(os.path.abspath(__file__))[0]
        cfg = configparser.ConfigParser()
        if ( os.path.exists( os.path.join(thisDir,'pycrtm_','pycrtm_setup.txt') ) ):
            pycrtm_setup_dir = os.path.join(thisDir,'pycrtm_','pycrtm_setup.txt')
        else:
            print("Error. File not present: {}".format(os.path.join(thisDir,'pycrtm_','pycrtm_setup.txt')))
            sys.exit()
        cfg.read( os.path.join(thisDir,pycrtm_setup_dir) )
        self.coefficientPath = cfg['Coefficients']['path_used']+"/"

        self.sensor_id = ''
        self.profiles = []
        self.traceConc = []
        self.traceIds = []
        self.usedGases = []
        self.Bt = []
        self.TauLevels = []
        self.surfEmisRefl = []
        self.TK = []
        self.QK = []
        self.O3K = []
        self.CO2K = []
        self.N2OK = []
        self.CH4K = []
        self.COK = []
        self.SkinK = []
        self.SurfEmisK = []
        self.SurfReflK = []
        self.windSpeedK = []
        self.windDirectionK = []
        self.Wavenumbers = []
        self.wavenumbers = []
        self.wavenumber = []
        self.Wavenumber = []
        self.frequencyGHz = []
        self.wavelengthMicrons = []
        self.wavelengthCm = []
        self.channelSubset = []
        self.subsetOn = False
        self.nChan = 0
        self.nChan_jacobian = 0
        self.output_tb_flag = True
        self.Active = False
        self.output_cloud_K = False
        self.output_aerosol_K = False
        self.StoreTrans = True
        self.StoreEmis = True
        self.nThreads = 1
        self.Reflectivity = []
        self.ReflectivityAttenuated = []
        self.CloudNcBin = 'Binary'
        self.AerosolNcBin = 'Binary'
        self.CoefNcBin = 'Binary'
        self.MWwaterCoeff_File = 'FASTEM6.MWwater.EmisCoeff.bin'
        self.IRwaterCoeff_File = 'Nalli.IRwater.EmisCoeff.bin'
        self.AerosolCoeff_File = 'AerosolCoeff.bin'
        self.CloudCoeff_File = 'CloudCoeff.bin'
        if(not os.path.exists(os.path.join(self.coefficientPath,self.MWwaterCoeff_File))):
            self.MWwaterCoeff_File = 'FASTEM6.MWwater.EmisCoeff.nc'
            self.CloudNcBin = 'netCDF' 
        if(not os.path.exists(os.path.join(self.coefficientPath,self.IRwaterCoeff_File))):
            self.IRwaterCoeff_File = 'Nalli.IRwater.EmisCoeff.nc'
            self.CloudNcBin = 'netCDF' 
        if(not os.path.exists(os.path.join(self.coefficientPath,self.AerosolCoeff_File))):
            self.IRwaterCoeff_File = 'AerosolCoeff.nc'
            self.CloudNcBin = 'netCDF' 
        if(not os.path.exists(os.path.join(self.coefficientPath,self.CloudCoeff_File))):
            self.IRwaterCoeff_File = 'CloudCoeff.nc'
            self.CloudNcBin = 'netCDF' 
 
        
        self.Height = []
        self.output_attenuated = True # logical to flip reflectivity in active sensor mode
        self.Algorithm = RT_ADA
    def loadInst(self):
        binPath = os.path.join(self.coefficientPath, self.sensor_id+'.SpcCoeff.bin')
        ncPath = os.path.join(self.coefficientPath, self.sensor_id+'.SpcCoeff.nc')
        if ( os.path.exists( binPath )  ):
            try:
                o = readSpcCoeff( binPath )
            except:
                print('Wrong Endian Binary Found.')
                if(not os.path.exists(ncPath)):
                    print('Problem. No Coefficients Valid (litte endian) here: {}'.format(self.coefficientPath))
                    sys.exit()
        if ( os.path.exists( ncPath ) ):
            print('Using netCDF.') 
            o = readSpcCoeffNc( ncPath )
            self.CoefNcBin = 'netCDF'
        if ( os.path.exists(ncPath) or os.path.exists(binPath)):
            self.nChanTotal = len(o['Wavenumber']) 
            self.channelSubset = np.arange(self.nChanTotal,dtype=np.int16)+1
            # For those who care to associate channel number with something physical:
            # just to save sanity put the permutations of (W/w)avenumber(/s) in here so things just go.
            self.wavenumbers = np.asarray(o['Wavenumber'])
            self.wavenumber = self.wavenumbers
            self.Wavenumber = self.wavenumbers 
            self.Wavenumbers = self.wavenumbers
            #For those more microwave oriented:
            self.frequencyGHz = 29.9792458 * self.wavenumbers
            self.wavelengthCm = 1.0/self.wavenumbers
            # And those who aren't interferometer oriented (people who like um): 
            self.wavelengthMicrons = 10000.0/self.wavenumbers
            self.wmo_sensor_id = o['wmo_sensor_id']
            self.wmo_satellite_id = o['wmo_satellite_id']
            if o['Sensor_Type'] == 101:
                self.Active = True
            if (self.AerosolCoeff_File.split('.')[-1] != 'bin'):
                self.AerosolNcBin = 'netCDF'
            
            if (self.CloudCoeff_File.split('.')[-1] != 'bin'):
                self.CloudNcBin = 'netCDF'
        else:
            print("Warning! {} doesn't exist!".format( os.path.join(self.coefficientPath, self.sensor_id+'.SpcCoeff.bin') ) )        
            print("Warning! {} doesn't exist!".format( os.path.join(self.coefficientPath, self.sensor_id+'.SpcCoeff.nc') ) )        
    def setupGases(self):

        #If this has been run by previous call to runK or runDirect, don't run it again!
        if(len(self.traceIds)>0): return

        # Figure out what gases the user has defined in profile
        availableGases = list(gases.keys())
        profileItems = list(self.profiles._asdict().keys())
        for p in profileItems:
            if (p in availableGases):self.usedGases.append(p)

        #Set the size of the trace gas array. 
        max_abs = len(self.usedGases)
        nprof, nlay = self.profiles.T.shape 
        self.traceConc = np.zeros([nprof,nlay,max_abs])
        self.traceIds = np.zeros(max_abs, dtype=int)

        #Fill array with what the user specified in profile.
        for i,g in enumerate(self.usedGases):
            self.traceConc[:,:,i] = self.profiles._asdict()[g][:,:]
            self.traceIds[i] = gases[g]
    def setupSubset(self):
        pyIdx = self.channelSubset - 1
        self.wavenumbers = self.wavenumbers[pyIdx] 
        self.wavenumber = self.wavenumber[pyIdx]
        self.Wavenumber = self.Wavenumber[pyIdx]
        self.Wavenumbers = self.Wavenumbers[pyIdx]
        #For those more microwave oriented:
        self.frequencyGHz = 29.9792458 * self.wavenumbers
        self.wavelengthCm = 1.0/self.wavenumbers
        # And those who aren't interferometer oriented (people who like um): 
        self.wavelengthMicrons = 10000.0/self.wavenumbers
        self.subsetOn= True            
    def runDirect(self):
        if(not len(self.surfEmisRefl)==0):
            pycrtm.emissivityreflectivity =  np.asfortranarray(self.surfEmisRefl)
            use_passed = True
        else: use_passed = False    
        #print(pycrtm.wrap_forward.__doc__)
        self.setupGases() 
        if('aerosolType' in list(self.profiles._asdict().keys())): 
            pycrtm.aerosoltype = self.profiles.aerosolType
            pycrtm.aerosoleffectiveradius = self.profiles.aerosols[:,:,:,1]
            pycrtm.aerosolconcentration = self.profiles.aerosols[:,:,:,0]
        if('cloudType' in list(self.profiles._asdict().keys())):
            pycrtm.cloudtype = self.profiles.cloudType
            pycrtm.cloudeffectiveradius = self.profiles.clouds[:,:,:,1]
            pycrtm.cloudconcentration = self.profiles.clouds[:,:,:,0]
            pycrtm.cloudfraction =  self.profiles.cloudFraction
        # setup stuff for channel subsetting
        self.channelSubset = np.asarray(self.channelSubset)
        if( not self.subsetOn and  self.channelSubset.shape[0] != self.nChanTotal):
            self.nChan = self.channelSubset.shape[0]
            self.setupSubset()
        if(not self.subsetOn):
            self.nChan = self.nChanTotal
        else:
            self.nChan = self.channelSubset.shape[0]
        if(self.Active == False): 
            self.Bt = pycrtm.wrap_forward( self.coefficientPath,
                                           self.Algorithm, 
                                           self.sensor_id,
                                           self.channelSubset,
                                           self.subsetOn,
                                           self.AerosolCoeff_File,
                                           self.CloudCoeff_File, 
                                           self.IRwaterCoeff_File,
                                           self.MWwaterCoeff_File, 
                                           self.output_tb_flag,
                                           self.StoreTrans,
                                           self.CloudNcBin ,
                                           self.AerosolNcBin,
                                           self.CoefNcBin,
                                           self.profiles.Angles[:,0], 
                                           self.profiles.Angles[:,4], 
                                           self.profiles.Angles[:,1], 
                                           self.profiles.Angles[:,2:4],
                                           self.profiles.SurfGeom[:,0], 
                                           self.profiles.SurfGeom[:,1], 
                                           self.profiles.SurfGeom[:,2], 
                                           self.StoreEmis,
                                           use_passed,
                                           self.profiles.DateTimes[:,0],
                                           self.profiles.DateTimes[:,1],
                                           self.profiles.DateTimes[:,2],
                                           self.profiles.Pi, 
                                           self.profiles.P, 
                                           self.profiles.T,
                                           self.traceConc,
                                           self.traceIds,
                                           self.profiles.climatology,
                                           self.profiles.surfaceTemperatures, 
                                           self.profiles.surfaceFractions, 
                                           self.profiles.LAI, 
                                           self.profiles.Salinity, 
                                           self.profiles.windSpeed10m, 
                                           self.profiles.windDirection10m,
                                           self.profiles.surfaceTypes[:,0], 
                                           self.profiles.surfaceTypes[:,1], 
                                           self.profiles.surfaceTypes[:,2], 
                                           self.profiles.surfaceTypes[:,3], 
                                           self.profiles.surfaceTypes[:,4], 
                                           self.profiles.surfaceTypes[:,5], 
                                           self.nThreads )
        elif(self.Active):
            
            #print(pycrtm.wrap_forward_active.__doc__) 
            self.StoreTrans = False
            self.Reflectivity, self.ReflectivityAttenuated, self.Height = pycrtm.wrap_forward_active( self.coefficientPath,
                                                                                          self.sensor_id,
                                                                                          self.channelSubset,
                                                                                          self.subsetOn,
                                                                                          self.AerosolCoeff_File,
                                                                                          self.CloudCoeff_File,
                                                                                          self.IRwaterCoeff_File,
                                                                                          self.MWwaterCoeff_File,
                                                                                          self.CloudNcBin ,
                                                                                          self.AerosolNcBin,
                                                                                          self.CoefNcBin,
                                                                                          self.profiles.Angles[:,0],
                                                                                          self.profiles.Angles[:,4],
                                                                                          self.profiles.Angles[:,1],
                                                                                          self.profiles.Angles[:,2:4],
                                                                                          self.profiles.SurfGeom[:,0],
                                                                                          self.profiles.SurfGeom[:,1],
                                                                                          self.profiles.SurfGeom[:,2],
                                                                                          self.StoreEmis,
                                                                                          use_passed,
                                                                                          self.profiles.DateTimes[:,0],
                                                                                          self.profiles.DateTimes[:,1],
                                                                                          self.profiles.DateTimes[:,2],
                                                                                          self.profiles.Pi,
                                                                                          self.profiles.P,
                                                                                          self.profiles.T,
                                                                                          self.traceConc,
                                                                                          self.traceIds,
                                                                                          self.profiles.climatology,
                                                                                          self.profiles.surfaceTemperatures,
                                                                                          self.profiles.surfaceFractions,
                                                                                          self.profiles.LAI,
                                                                                          self.profiles.Salinity,
                                                                                          self.profiles.windSpeed10m,
                                                                                          self.profiles.windDirection10m,
                                                                                          self.profiles.surfaceTypes[:,0],
                                                                                          self.profiles.surfaceTypes[:,1],
                                                                                          self.profiles.surfaceTypes[:,2],
                                                                                          self.profiles.surfaceTypes[:,3],
                                                                                          self.profiles.surfaceTypes[:,4],
                                                                                          self.profiles.surfaceTypes[:,5],
                                                                                          self.nThreads )        
        if(self.StoreTrans):
            self.TauLevels = pycrtm.outtransmission
        if(self.StoreEmis):
            self.surfEmisRefl = pycrtm.emissivityreflectivity
    def runK(self):
        if(not len(self.surfEmisRefl)==0):
            pycrtm.emissivityreflectivity = np.asfortranarray(self.surfEmisRefl)
            use_passed=True
        else: use_passed=False                  
        self.setupGases() 
       
        # print(pycrtm.wrap_k_matrix.__doc__) 
        if('aerosolType' in list(self.profiles._asdict().keys())): 
            pycrtm.aerosoltype = self.profiles.aerosolType
            pycrtm.aerosoleffectiveradius = self.profiles.aerosols[:,:,:,1]
            pycrtm.aerosolconcentration = self.profiles.aerosols[:,:,:,0]
        if('cloudType' in list(self.profiles._asdict().keys())):
            pycrtm.cloudtype = self.profiles.cloudType
            pycrtm.cloudeffectiveradius = self.profiles.clouds[:,:,:,1]
            pycrtm.cloudconcentration = self.profiles.clouds[:,:,:,0]
            pycrtm.cloudfraction =  self.profiles.cloudFraction
        self.channelSubset = np.asarray(self.channelSubset)
        if( not self.subsetOn and  self.channelSubset.shape[0]!= self.nChanTotal):
            self.setupSubset()
        self.nChan_jac = 0
        if(not self.subsetOn):
            self.nChan = self.nChanTotal
        else:
            self.nChan = self.channelSubset.shape[0]
        if(self.output_cloud_K):
            cld_dims = self.profiles.clouds.shape
            self.nChan_jacobian = self.nChan
        else:
            cld_dims = [0,0,0] 
        if(self.output_aerosol_K):
            aer_dims = self.profiles.aerosols.shape
            self.nChan_jacobian = self.nChan
        else:
            aer_dims = [0,0,0]
         
        jac_1_dim = max(aer_dims[0],cld_dims[0])
        jac_2_dim = max(aer_dims[1],cld_dims[1])
        if (self.Active == False):
            self.Bt, self.TK, traceK, self.SkinK, self.SurfEmisK, self.ReflK,self.WindSpeedK, self.windDirectionK,\
            self.CloudEffectiveRadiusK, self.CloudConcentrationK, self.CloudFractionK,\
            self.AerosolEffectiveRadiusK, self.AerosolConcentrationK                                           =  pycrtm.wrap_k_matrix(  self.coefficientPath,
                                                                                                                                         self.Algorithm,
                                                                                                                                         self.sensor_id,
                                                                                                                                         self.channelSubset,
                                                                                                                                         self.subsetOn,
                                                                                                                                         self.AerosolCoeff_File,
                                                                                                                                         self.CloudCoeff_File, 
                                                                                                                                         self.IRwaterCoeff_File,
                                                                                                                                         self.MWwaterCoeff_File,
                                                                                                                                         self.output_tb_flag,
                                                                                                                                         self.StoreTrans,
                                                                                                                                         self.output_cloud_K,
                                                                                                                                         self.output_aerosol_K,
                                                                                                                                         self.CloudNcBin,
                                                                                                                                         self.AerosolNcBin,
                                                                                                                                         self.CoefNcBin,
                                                                                                                                         self.profiles.Angles[:,0], 
                                                                                                                                         self.profiles.Angles[:,4], 
                                                                                                                                         self.profiles.Angles[:,1], 
                                                                                                                                         self.profiles.Angles[:,2:4],
                                                                                                                                         self.profiles.SurfGeom[:,0], 
                                                                                                                                         self.profiles.SurfGeom[:,1], 
                                                                                                                                         self.profiles.SurfGeom[:,2], 
                                                                                                                                         self.StoreEmis,  
                                                                                                                                         use_passed, 
                                                                                                                                         self.profiles.DateTimes[:,0], 
                                                                                                                                         self.profiles.DateTimes[:,1],
                                                                                                                                         self.profiles.DateTimes[:,2],
                                                                                                                                         self.nChan_jacobian,
                                                                                                                                         jac_1_dim,
                                                                                                                                         jac_2_dim,
                                                                                                                                         cld_dims[2],
                                                                                                                                         aer_dims[2],
                                                                                                                                         self.profiles.Pi, 
                                                                                                                                         self.profiles.P, 
                                                                                                                                         self.profiles.T, 
                                                                                                                                         self.traceConc, 
                                                                                                                                         self.traceIds,
                                                                                                                                         self.profiles.climatology,
                                                                                                                                         self.profiles.surfaceTemperatures, 
                                                                                                                                         self.profiles.surfaceFractions, 
                                                                                                                                         self.profiles.LAI, 
                                                                                                                                         self.profiles.Salinity, 
                                                                                                                                         self.profiles.windSpeed10m, 
                                                                                                                                         self.profiles.windDirection10m,
                                                                                                                                         self.profiles.surfaceTypes[:,0], 
                                                                                                                                         self.profiles.surfaceTypes[:,1], 
                                                                                                                                         self.profiles.surfaceTypes[:,2], 
                                                                                                                                         self.profiles.surfaceTypes[:,3], 
                                                                                                                                         self.profiles.surfaceTypes[:,4], 
                                                                                                                                         self.profiles.surfaceTypes[:,5], 
                                                                                                                                         self.nThreads )
        elif(self.Active):
            self.Height, self.Reflectivity, self.ReflectivityAttenuated, self.TK, traceK,\
            self.CloudEffectiveRadiusK, self.CloudConcentrationK, self.CloudFractionK,\
            self.AerosolEffectiveRadiusK, self.AerosolConcentrationK                                           =  pycrtm.wrap_k_matrix_active(  self.coefficientPath,
                                                                                                                                         self.sensor_id,
                                                                                                                                         self.channelSubset,
                                                                                                                                         self.subsetOn,
                                                                                                                                         self.AerosolCoeff_File,
                                                                                                                                         self.CloudCoeff_File, 
                                                                                                                                         self.IRwaterCoeff_File,
                                                                                                                                         self.MWwaterCoeff_File,
                                                                                                                                         self.output_attenuated,
                                                                                                                                         self.output_cloud_K,
                                                                                                                                         self.output_aerosol_K,
                                                                                                                                         self.CloudNcBin,
                                                                                                                                         self.AerosolNcBin,
                                                                                                                                         self.CoefNcBin,
                                                                                                                                         self.profiles.Angles[:,0], 
                                                                                                                                         self.profiles.Angles[:,4], 
                                                                                                                                         self.profiles.Angles[:,1], 
                                                                                                                                         self.profiles.Angles[:,2:4],
                                                                                                                                         self.profiles.SurfGeom[:,0], 
                                                                                                                                         self.profiles.SurfGeom[:,1], 
                                                                                                                                         self.profiles.SurfGeom[:,2], 
                                                                                                                                         self.profiles.DateTimes[:,0], 
                                                                                                                                         self.profiles.DateTimes[:,1],
                                                                                                                                         self.profiles.DateTimes[:,2],
                                                                                                                                         self.nChan_jacobian,
                                                                                                                                         jac_1_dim,
                                                                                                                                         jac_2_dim,
                                                                                                                                         cld_dims[2],
                                                                                                                                         aer_dims[2],
                                                                                                                                         self.profiles.Pi, 
                                                                                                                                         self.profiles.P, 
                                                                                                                                         self.profiles.T, 
                                                                                                                                         self.traceConc, 
                                                                                                                                         self.traceIds,
                                                                                                                                         self.profiles.climatology,
                                                                                                                                         self.profiles.surfaceTemperatures, 
                                                                                                                                         self.profiles.surfaceFractions, 
                                                                                                                                         self.profiles.LAI, 
                                                                                                                                         self.profiles.Salinity, 
                                                                                                                                         self.profiles.windSpeed10m, 
                                                                                                                                         self.profiles.windDirection10m,
                                                                                                                                         self.profiles.surfaceTypes[:,0], 
                                                                                                                                         self.profiles.surfaceTypes[:,1], 
                                                                                                                                         self.profiles.surfaceTypes[:,2], 
                                                                                                                                         self.profiles.surfaceTypes[:,3], 
                                                                                                                                         self.profiles.surfaceTypes[:,4], 
                                                                                                                                         self.profiles.surfaceTypes[:,5], 
                                                                                                                                         self.nThreads )
 
        for i,ids in enumerate(list(self.traceIds)):
            # I think I can do something smarter here in python to contruct self.QK etc through an execute, or something along those lines?
            if(ids == gases['Q']):   self.QK   = traceK[:,:,:,i]
            if(ids == gases['O3']):  self.O3K  = traceK[:,:,:,i]
            if(ids == gases['CH4']): self.CH4K = traceK[:,:,:,i]
            if(ids == gases['CO2']): self.CO2K = traceK[:,:,:,i]
            if(ids == gases['CO']):  self.COK  = traceK[:,:,:,i]
            if(ids == gases['N2O']): self.N2OK = traceK[:,:,:,i]
        # if we don't have any "weird" gases, empty out traceK,traceConc to save on RAM.
        if not any(g in self.usedGases for g in  ['Q', 'O3', 'CH4', 'CO','CO2', 'N2O']):
            print("saving on RAM")
            self.traceK = []
            self.traceConc = []     
      
        if(self.StoreTrans):
            self.TauLevels = pycrtm.outtransmission

        if(self.StoreEmis):
            self.surfEmisRefl = pycrtm.emissivityreflectivity
            
