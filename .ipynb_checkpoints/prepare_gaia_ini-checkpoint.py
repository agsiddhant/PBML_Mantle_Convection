import os

def create_ini_file(f_gaia_ini, mode, raq, fkt, fkp, advection_scheme, 
                    intervene_TS, warm_up_steps, solver="mumps", initialization="hot", 
                    urf=1, Di=0, core_cool=False, radioactive_decay=False, CaseID="case"):
    
    start = [      "GridFile		=	CREATE \n", 
                   "BOX/Layers		=	126 \n", 
                   "BOX/AspectRatio	=	4 \n", 
                   "BOX/Dimensions	=	2 \n", 
                   "Restart			=	no \n"
                  ]

    time = [      "MaxTime			      =	10 \n", 
                  "InitialDT		      =	1e-7 \n", 
                  "MaxDT			      =	1e-4 \n", 
                  "TSType			      =	COURANT \n", 
                  "TSFactor		          =	1  \n", 
                  "SteadyState/Threshold  = 	1e-3 \n", 
                  "SteadyState/Value	  = 	1 \n", 
                  ]
    
    out =       ["CaseID			=	" + CaseID + " \n", 
                 "SnapshotIter	    =	10000000000000000000000 \n", 
                 "OutputIter		=	1000000000000000000000 \n", 
                 "OutputTime		=	0. \n", 
                 "OutputType		=	TSPVv \n" 
                 ]

    modules =   ["MCInit          	=	Box/Init, InitSphHarmonics ", 
                 "MCBody		    =	Boussinesq \n", 
                 "MCPreTS         	=	 \n", 
                 "MCPostOuter     	=    \n", 
                 "MCPrePressure   	=    \n", 
                 "MCPostTS       	=	SteadyState \n", 
                 "MCEnergy        	=	Boussinesq \n", 
                 "MCRheology      	=	FKViscosity \n", 
                 "MCPreOutput     	=   \n", 
                 "MCOutput        	=	\n"
                ]

    if Di>0:
        modules[6] = "MCEnergy        	=	Boussinesq/Compress \n"

    if initialization == "linear":
        modules[0] +=  ", InitTempLinear "
    elif initialization == "perfect":
        modules[0] += ", ReadASCII "

    if core_cool and not radioactive_decay:
        modules[5] = "MCPostTS       	=	Core/Cooling \n" 
        modules[0] += ", Core/Init " 
    elif not core_cool and radioactive_decay:
        modules[5] = "MCPostTS       	=	RadioactiveDecay \n" 
        modules[0] += ", RadioactiveDecay/Init " 
    elif core_cool and radioactive_decay:
        modules[5] =  "Core/Cooling, RadioactiveDecay \n"
        modules[0] += ", Core/Init, RadioactiveDecay/Init " 
    
    modules[0] +=  " \n"
    
    evolution    = ["RadioactiveDecay/nDecay     =   4                   \n",	 
                    "RadioactiveDecay/Lambda0   =   14.200767386369366  \n",
                    "RadioactiveDecay/Coeff0	=	0.130448695228009   \n",
                    "RadioactiveDecay/Lambda1   =   90.1668042856123    \n",
                    "RadioactiveDecay/Coeff1	=	0.2345333106414419  \n",
                    "RadioactiveDecay/Lambda2   =   4.534102158362219   \n",
                    "RadioactiveDecay/Coeff2	=	0.07981198571490902 \n",
                    "RadioactiveDecay/Lambda3   =   50.78194417365685   \n", 
                    "RadioactiveDecay/Coeff3	=	0.55520600841564    \n",
                    "Core/rhoCpVar              =   0.7058823529411765  \n"
                    ]

    

    init_temp = 0 if initialization == "cold" else 1
    case  = [#"InitialTemp		 =	" + str(init_temp) + "  \n", 	
             "InitialTemperature =	" + str(init_temp) + "  \n", 	
             "InitialModeL		 =	-1	  \n", 			
             "InitialModeM		 =	-1  \n", 
             "InitialAmp		 =	0.01  \n", 
             "ReadASCII/Field/T  = ml_prof.txt   \n" 
            ]

    bc  = ["BCBottomVisc		    =	0 	  \n", 		
            "BCTopVisc		        =	0		  \n", 	
            "BCBottomHFlow		    =	no	  \n", 		
            "BCBottomHValue  	    =	1	  \n", 		
            "BCTopHFlow		        =	no		  \n", 	
            "BCTopHValue  		    =	0	  \n", 		
            "ITL/TopLayerDepth  	=	0.05  \n", 	
            "ITL/TopLayerMax	  	= 	0.75  \n", 	
            "ITL/BottomLayerDepth   =	0.95  \n", 	
            "ITL/BottomLayerMin 	=	0.75 \n" 
          ]

    paras = ["Ra                      =       1e0 \n", 	
             "RaQ                     =       " + str(raq) + " \n", 
             "FKViscosity/ViscT       =       " + str(fkt) + " \n", 
             "FKViscosity/ViscP       =       " + str(fkp) + " \n", 
             "Di                      =       " + str(Di) + " \n", 
             "PrInverted		      =       0   \n", 
             "Tref                    =       0   \n", 
             "Dref                    =       0   \n", 
             "T0			          =	      0   \n", 
             ]
                
    numerics = ["Debug			      =	2 \n", 	
                "IterLimitOuter		  =	1 \n", 	
                "Advection		      =	" + str(advection_scheme) + " \n"
                "ViscosityStabilizer  = 	0 \n", 	
                "MMSolverSkip         = " + str(intervene_TS)  + " \n", 	 
                "MMSolverSkipWarmUp   = " + str(warm_up_steps) + " \n",
                "@ini/lineout.ini  \n",
                "LineOut/OutputEveryN = 10  \n"
               ]

    if solver=="mumps":
        numerics += ["MMSolver		      =	MUMPS \n", 	
                     "MUMPS/ICNTL_7		  =	4 \n", 	
                     "FixPressure		  =	7707 \n", 	
                    ]
    else:
        numerics += ["urf_mm			      =	" + str(urf) + " \n"]


    lines = start + time + out + modules + case + bc + evolution + paras + numerics 
    file = open(f_gaia_ini, "w")
    file.writelines(lines)
    file.close()