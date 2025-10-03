
def time2depth(tsection, vrmsmodel, tt):

    """
    Description: 
       Convert a  time-migrated section or a trace to a depth section, or a trace 
       using a RMS-velocity model (scalar or vector)
                
    Use:
       [zmigsection,zz] = time2depth(tmigsection,vrmsmodel,tt)

    Parameters (input):
       tsection:  1D np.ndarray (trace) or 2D np.ndarray (section)
                   Seismic trace or section in time domain
       vrmsmodel: scalar, or np.ndarray of the same shape as tsection
                   RMS-velocity, 3 options:
                   1) constant velocity : float or int
                      single velocity value for the entire section or a trace
                   2) 1-D velocity vector : 
                      velocity vector for depth conversion of a single trace
                   3) 2-D velocity vector : 
                      velocity matrix for depth conversion of a seismic section                    
       tt: 1D np.ndarray of the same length a tsection.shape[0]
           Time vector as 1D np.ndarray

    Returns (output):
       zsection : depth-converted time-migrated section or a single trace
       zz : depth vector

    """
    if len(tsection.shape) == 2:
    
        def time2depth_section_constVrms(tsection, vrms, tt):

            """
            Description: 
               Convert a  time-migrated section to a depth section, using a RMS-velocity model

             Usage:
                 [zmigsection,zz] = time2depth_SECTION(tmigsection,vrmsmodel,tt)

             Parameters (input):
                tsection : 2D np.ndarray
                           Time (possibly time-migrated) section
                vrms : float or int
                       Constant RMS-velocity
                tt : 1D np.ndarray of the same size as zsection.shape[0]
                      Time vector as 1D np.ndarray

             Returns (output):
                zsection : depth-converted time-migrated section
                zz       : depth vector

            """

            dt = tt[1] - tt[0]
            nt = len(tt)
            nx = tsection.shape[1]
            vintmodel = np.full((nt,nx), vrms)

            dt = tt[1] - tt[0]
            nt = len(tt)
            nx = tsection.shape[1]

            # take dz as velocity times dt/2 (two-way time):
            dz = vrms*dt/2

            # take maximum depth as velocity times tmax/2 (two-way time):
            tmax = tt[len(tt)-1]
            zmax = vrms*tmax/2
            nz = int(np.ceil(zmax/dz+1))
            zmax2 = nz*dz
            zz = np.arange(dz, zmax2+dz, dz)

            print(' ')
            print(' --- Total number of traces to be converted to depth: ' + str(nx) + ' --- ')
            print(' ')

            # now we need to interpolate to regular np.range(dz, zmax) with dz step
            zsection = np.zeros((nz, nx))

            printcounter = 0
            tenPerc = int(nx/10)
            percStatus = 0

            for ix in range(0,nx):

                zsection[0,ix] = tsection[0,ix]
                itrun          = 0
                z1             = 0.0
                z2             = zmax2

                for iz in range(1,int(nz)):

                    ztrue = iz*dz

                    # find out between which time samples are needed for interpolation:
                    if itrun < nt:
                        z2 = z1 + (vintmodel[itrun-1, ix]*dt/2)
                        while ztrue > z2 and itrun < nt:
                            itrun = itrun +1
                            z1    = z2
                            z2    = z2 + vintmodel[itrun-1,ix]*dt/2

                        if itrun < nt:
                            zsection [iz, ix] = (z2-ztrue)/(z2-z1)*tsection[itrun-1, ix] + (ztrue-z1)/(z2-z1)*tsection[itrun, ix]

                if printcounter == tenPerc:
                    percStatus += 10
                    print('Finished depth converting {} traces out of {}. {}%'.format(ix, nx, percStatus))
                    printcounter=0
                printcounter+=1
            print('Done!')

            return zsection, zz


        def time2depth_section_modelVrms(tsection, vrmsmodel, tt):
            """
            Description: 
               Convert a  time-migrated section to a depth section, using a RMS-velocity model 

            Usage:
               [zmigsection,zz] = time2depth_section_modelVrms(tmigsection,vrmsmodel,tt)

            Parameters (input):
               tsection : time (possibly time-migrated) section
                vmodel  : RMS-velocity model
                tt      : time vector

            Returns (output):
               zsection : depth-converted time-migrated section
               zz       : depth vector

            """

            dt = tt[1] - tt[0]
            nt = len(tt)
            nx = tsection.shape[1]

            vintmodel       = np.zeros((nt,nx))
            ix              = 0
            vintmodel[0,ix] =  vrmsmodel[0,ix]

            for it in range(1, nt):
                v2diff = tt[it]*vrmsmodel[it, ix]**2 - tt[it-1]*vrmsmodel[it-1, ix]**2
                vintmodel[it, ix] = np.sqrt(v2diff/dt)

            for ix in range(1,nx):
                vintmodel[0,ix] = vrmsmodel[0,ix]
                for it in range(1,nt):
                    v2diff = tt[it]*vrmsmodel[it,ix]**2 - tt[it-1]*vrmsmodel[it-1,ix]**2
                    vintmodel[it,ix] = np.sqrt(v2diff/dt)

            # determine minimum velocity for minimum sampling in depth z
            vrmsmin = np.min(vrmsmodel)

            # take dz as smallest velocity times dt/2 (two-way time):
            dz = vrmsmin*dt/2

            # take maximum depth as maximum velocity times tmax/2 (two-way time):
            tmax = tt[len(tt)-1]
            vrmsmax = np.max(vrmsmodel)
            zmax = vrmsmax*tmax/2
            nz = int(np.ceil(zmax/dz+1))
            zmax2 = nz*dz
            zz = np.arange(dz, zmax2+dz, dz)

            print(' ')
            print(' --- Total number of traces to be converted to depth: ' + str(nx) + ' --- ')
            print(' ')

            # now we need to interpolate to regular np.range(dz, zmax) with dz step
            zsection = np.zeros((nz, nx))

            printcounter = 0
            tenPerc = int(nx/10)
            percStatus = 0

            for ix in range(0,nx):

                zsection[0,ix] = tsection[0,ix]
                itrun          = 0
                z1             = 0.0
                z2             = zmax2

                for iz in range(1,int(nz)):

                    ztrue = iz*dz

                    # find out between which time samples are needed for interpolation:
                    if itrun < nt:
                        z2 = z1 + (vintmodel[itrun-1, ix]*dt/2)
                        while ztrue > z2 and itrun < nt:
                            itrun = itrun +1
                            z1    = z2
                            z2    = z2 + vintmodel[itrun-1,ix]*dt/2

                        if itrun < nt:
                            zsection [iz, ix] = (z2-ztrue)/(z2-z1)*tsection[itrun-1, ix] + (ztrue-z1)/(z2-z1)*tsection[itrun, ix]

                if printcounter == tenPerc:
                    percStatus += 10
                    print('Finished depth converting {} traces out of {}. {}%'.format(ix, nx, percStatus))
                    printcounter=0

                printcounter+=1

            print('Done!')

            return zsection, zz


        def time2depth_section(tsection, vrmsmodel, tt):

            """
            time2depth: Convert a  time-migrated section to a depth section, 
                        using a RMS-velocity model (scalar or vector)

            Usage:
                 [zmigsection,zz] = time2depth_section(tmigsection,vrmsmodel,tt)

             Returns (output):
                zsection : depth-converted time-migrated section
                zz       : depth vector
             Returns (output):
                ztrace : depth-converted trace
                zz     : depth vector

             Parameters (input):
                tsection : 2D np.ndarray
                           Seismic section in time domain
                vrmsmodel : scalar or 2D np.ndarray of the same shape as tsection
                            RMS-velocity, two options:
                            1) constant velocity : single velocity value for the entire section, 
                                                   float or int
                            2) 1-D velocity vector : velocity matrix
                tt : 1D np.ndarray of the same length as tsection.shape[0]
                     Time vector as 1D np.ndarray

            """

            if type (vrmsmodel) != np.ndarray:
                ztrace, zz = time2depth_section_constVrms (tsection, vrmsmodel, tt)
                print('Input section has been depth-converted using constant velocity.')

            elif type (vrmsmodel) == np.ndarray:
                ztrace, zz = time2depth_section_modelVrms(tsection, vrmsmodel, tt)
                print('Input section has been depth-converted using variable velocity model.')

            # else: 
            #     print ('Unsupported data type for the input velocity! Please provide the input velocity in a 1D numpy array or as a scalar.')

            return ztrace, zz

        if type (vrmsmodel) != np.ndarray:
            ztrace, zz = time2depth_section_constVrms (tsection, vrmsmodel, tt)
            print('Input section has been depth-converted using constant velocity.')

        elif type (vrmsmodel) == np.ndarray:
            ztrace, zz = time2depth_section_modelVrms(tsection, vrmsmodel, tt)
            print('Input section has been depth-converted using variable velocity model.')
            
            
    if len(tsection.shape) == 1:
        
        def time2depth_trace_constVrms (ttrace, vrms, tt):
            
            """
            Description:
               Convert a  single trace in the time domain to the depth domain
               using a constant RMS-velocity

             Usage:
                [ztrace,zz] = time2depth_trace_constVrms(ttrace,vrmsmodel,tt)

             Parameters (input):
                ttrace : 1D np.ndarray
                          Trace in time domain
                vrms : float or int
                       Constant RMS-velocity
                tt : 1D np.ndarray of the same length as ttrace
                     Time vector as 1D np.ndarray

             Returns (output):
                ztrace : depth-converted trace
                zz     : depth vector

            """

            dt = tt[1] - tt[0]
            nt = len(tt)

            vintmodel = np.full(nt, vrms)

            # take dz as  velocity times dt/2 (two-way time):
            dz = vrms*dt/2

            # take maximum depth as velocity times tmax/2 (two-way time):
            tmax = tt[-1]
            zmax = vrms*tmax/2

            nz = int(np.ceil(zmax/dz+1))
            zmax2 = nz*dz
            zz = np.arange(dz, zmax2, dz)

            # now we need to interpolate to regular np.range(dz, zmax) with dz step

            ztrace = np.zeros((nz))
            ztrace[0]      = ttrace[0]
            itrun          = 0
            z1             = 0.0
            z2             = zmax2

            for iz in range(1,int(nz)):

                ztrue = iz*dz

                # find out between which time samples are needed for interpolation:
                if itrun < nt:
                    z2 = z1 + (vintmodel[itrun-1]*dt/2)
                    while ztrue > z2 and itrun < nt:

                        itrun = itrun +1
                        z1    = z2
                        z2    = z2 + vintmodel[itrun-1]*dt/2

                    if itrun < nt:
                        ztrace[iz] = (z2-ztrue)/(z2-z1)*ttrace[itrun-1] + (ztrue-z1)/(z2-z1)*ttrace[itrun]

            print('Done!')

            return ztrace, zz
        
        
        def time2depth_trace_modelVrms(ttrace, vrmsmodel, tt):
            """
            Description: 
               Convert a  single trace in the time domain to the depth domain
               using a RMS-velocity model

             Usage:
                [ztrace,zz] = time2depth_trace_modelVrms(ttrace,vrmsmodel,tt)

             Parameters (input):
                ttrace : 1D np.ndarray
                         Trace in time domain
                vmodel : 1D np.ndarray of the same length as ttrace
                         RMS-velocity
                tt : 1D np.ndarray of the same length as ttrace
                     Time vector as 1D np.ndarray

             Returns (output):
                ztrace : depth-converted trace
                zz     : depth vector

            """

            dt = tt[1] - tt[0]
            nt = len(tt)

            vintmodel    =  np.zeros(nt)
            vintmodel[0] =  vrmsmodel[0]

            for it in range(1, nt):
                v2diff = tt[it]*vrmsmodel[it]**2 - tt[it-1]*vrmsmodel[it-1]**2
                vintmodel[it] = np.sqrt(v2diff/dt)

            # determine minimum velocity for minimum sampling in depth z
            vrmsmin = np.min(vrmsmodel)

            # take dz as smallest velocity times dt/2 (two-way time):
            dz = vrmsmin*dt/2

            # take maximum depth as maximum velocity times tmax/2 (two-way time):
            tmax = tt[-1]
            vrmsmax = np.max(vrmsmodel)
            zmax = vrmsmax*tmax/2

            nz = int(np.ceil(zmax/dz+1))
            zmax2 = nz*dz
            zz = np.arange(dz, zmax2, dz)

            # now we need to interpolate to regular np.range(dz, zmax) with dz step

            ztrace = np.zeros((nz))
            ztrace[0]      = ttrace[0]
            itrun          = 0
            z1             = 0.0
            z2             = zmax2

            for iz in range(1,int(nz)):

                ztrue = iz*dz

                # find out between which time samples are needed for interpolation:
                if itrun < nt:
                    z2 = z1 + (vintmodel[itrun-1]*dt/2)
                    while ztrue > z2 and itrun < nt:

                        itrun = itrun +1
                        z1    = z2
                        z2    = z2 + vintmodel[itrun-1]*dt/2

                    if itrun < nt:
                        ztrace[iz] = (z2-ztrue)/(z2-z1)*ttrace[itrun-1] + (ztrue-z1)/(z2-z1)*ttrace[itrun]

            print('Done!')

            return ztrace, zz


        def time2depth_trace(ttrace, vrmsmodel, tt):
            """
            Description:
               Convert a  single trace in the time domain to the depth domain
               using a RMS-velocity model (scalar or vector)

             Usage:
                [ztrace,zz] = time2depth_trace(ttrace,vrmsmodel,tt)

             Parameters (input):
                ttrace : 1D np.ndarray
                         Trace in time domain
                vrmsmodel : scalar or 1D np.ndarray of the same length as ttrace
                            RMS-velocity, two options:
                            1) constant velocity   : single velocity value for ttrace, float
                            2) 1-D velocity vector : a velocity value for each row in ttrace, 1D np.ndarray o
                tt : 1D np.ndarray of the same length as ttrace
                     Time vector as 1D np.ndarray

             Returns (output):
                ztrace : depth-converted trace
                zz     : depth vector

            """
            if type (vrmsmodel) != np.ndarray:
                ztrace, zz = time2depth_trace_constVrms (ttrace, vrmsmodel, tt)
                print('Input trace has been depth-converted using constant velocity.')

            elif type (vrmsmodel) == np.ndarray:
                ztrace, zz = time2depth_trace_modelVrms(ttrace, vrmsmodel, tt)
                print('Input trace has been depth-converted using variable velocity model.')

            # else: 
            #     print ('Unsupported data type for the input velocity! Please provide the input velocity in a 1D numpy array or as a scalar.')

            return ztrace, zz

        if type (vrmsmodel) != np.ndarray:
            ztrace, zz = time2depth_trace_constVrms(tsection, vrmsmodel, tt)
            print('Input trace has been depth-converted using constant velocity.')

        elif type (vrmsmodel) == np.ndarray:
            ztrace, zz = time2depth_trace_modelVrms(tsection, vrmsmodel, tt)
            print('Input trace has been depth-converted using variable velocity model.')
        
    # else: 
    #     print ('Conversion unsuccessful. Please check the input time section / trace.')

    return ztrace, zz


