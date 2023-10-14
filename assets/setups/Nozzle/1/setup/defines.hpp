#pragma once
// ############################################ file paths #############################################

//#define PATH_DEBUG "/data2/s1molehm/FluidX3D/Bead/interpolation/"
//#define PATH_DEBUG "/tp6_gekle/data2/bt703473/debug/test2/"
#define PATH_DEBUG "./bin/debug/" // path for debug files (PTX, LOG, VALIDATION)
#define PATH_IBM   "./bin/ibm/"   // path for IBM extension


// ############################################ LBM options ############################################

// There hides a D2Q5 in the code
//#define D2Q9
//#define D3Q7 // very very inaccurate
//#define D3Q13 // very inaccurate; Only works with MRT according to Krüger book p. 87; Does not reproduce EQUILIBRIUM_STRESS correctly.

// Only use these:
//#define D3Q15
#define D3Q19 // Best value
//#define D3Q27 // Better for high Reynolds numbers.

#define SRT // singe relaxation time
//#define TRT // two relaxiation time (as fast as SRT, but more accurate)
//#define MRT // multi relaxiation time (as fast as SRT, but more accurate), (### not fully validated for for D3Q7 and D3Q13, incomplete relaxation times for D3Q7 and D3Q27 ###)

//#define UPDATE_RHO // update density field after every step (not necessary for pure LBM)
//#define UPDATE_U // update velocity field after every step (not necessary for pure LBM, makes tracer particle calculation faster)

//#define MOVING_WALLS // moving bounce-back boundaries (incompatible with EQUILIBRIUM_BOUNDARIES due to unnecessary flag recycling)
//#define EQUILIBRIUM_BOUNDARIES // non-reflecting velocity boundaries, generally more stable, but less accurate (incompatible with MOVING_WALLS due to unnecessary flag recycling)
//#define BOUNDARY_EXPRESSION // combinable with EQUILIBRIUM_BOUNDARIES xor MOVING_WALLS, !!!doesn't work on AMD for some reason!!!

#define VISCOELASTIC
#define EQUILIBRIUM_STRESS // Include viscoelastic forces by shifting the equilibrium populations (Onishi 2005) instead of via force
#define STRAIN_RATE_TENSOR_FROM_POP // If set, the strain rate tensor used for shuffling is calculated from the populations otherwise the one from the velocity gradient is used
// Only one of the following models may be defined
//#define OLDROYD_B
//#define FENEP
#define PTT // From Ferras 2019 eq (3) and (2) original PTT

//#define IBM // immersed boundary method (### under construction ###)
//#define IBM_CELLS // simulate RBCs with the immersed-boundary method
//#define INOUT // activate volume tracking for RBCs
//#define INOUT_CHANGE // mark change of inside/outside in second bit

// ############################################## Forces ###############################################

//#define TETRA  //TODO WORK IN PROGRESS
//#define MEMBRANE_FORCES //Incompatible with TETRA at the moment; MAY NOT USE TETRA FILES!!!
#define VOLUME_FORCE // volume force (global or with FORCE_FIELD stored in each individual LBM point) gets transfered into fluid. Theoretically conflicts with FORCE_ON_WALLS (using same array for different purpose), but is flagged in a way, that allows simultanious use. (can be turned off when not used, has no measurable performance impact)

// Force options

//#define IBM_INDIVIDUAL_REFERENCE  // ignores the reference curvature in provided in 'insert_cells'. Uses initial shape as reference. TODO replace with config option
#define FORCE_FIELD // Defines an array for a force to be stored in each LBM point. Is required by FORCE_ON_WALLS and optionally for VOLUME_FORCE
//#define FORCE_ON_WALLS // calculate force on walls and stores it in force field; Currently not working. Missing: call in fluid step (needs to be after virtual streaming, needs to reset F before IBM if applicable); Requires FORCE_FIELD to store output
//#define FORCE_EVALUATION // write sum of forces on wall nodes also flagged with TYPE_F to file (cpu side); Currently not working; Conflicts with VISCOELASTIC due to use of TYPE_F (which is completely unnecessary); Rework required

// ######################################### Additional Output #########################################
//#define STRAIN_RATE_TENSOR // computes the strain-rate tensor //TODO only do on output LBM steps
#define VISCOUS_STRESS_TENSOR // compute the visous stress tensor

// ######################################## computation options ########################################

//#define USE_64BIT_INDEXING // use 64-bit simulation box addressing (2% slower on GPUs, maximum box size for D3Q19: 32-bit: 608x608x608, 64-bit: 1616x1616x1616)
//#define THREAD_BLOCK_SIZE 256 // uncomment if you need a thread block size other than 256, best performance for 256, GPU warp size is 32, possible values: 32, 64, 128, 256, 512 //TODO not the same for all our GPUs

// ############################################# Utilities #############################################
#define UTILITIES_SERIALIZATION // Allows for certain object to be serialized and written to disk. Likely used in the currently unmaintaned checkpoint system

// ######################################## development options ########################################

//#define INDEX_CHECK // check if index is within expected range
//#define DEBUG  // enables extra debugging output  //TODO not functional DO NOT ENABLE!!!!!!!

// ########################################### define rules ############################################

#ifdef VISCOELASTIC
	#define VOLUME_FORCE
	#define FORCE_FIELD
	#define STRAIN_RATE_TENSOR
#else // !VISCOELASTIC
	#undef EQUILIBRIUM_STRESS
#endif // VISCOELASTIC

#ifdef FORCE_EVALUATION
	#define FORCE_ON_WALLS
#endif // FORCE_EVALUATION

#ifdef FORCE_ON_WALLS
	#define FORCE_FIELD
#endif // FORCE_ON_WALLS

#if defined(STRAIN_RATE_TENSOR) || defined(VISCOUS_STRESS_TENSOR)
	#define UPDATE_RHO
#endif // STRAIN_RATE_TENSOR || VISCOUS_STRESS_TENSOR

#ifdef IBM_CELLS
	#define IBM
#else // !IBM_CELLS
	#undef INOUT
#endif // IBM_CELLS

#ifdef IBM
	#define FORCE_FIELD
	#define VOLUME_FORCE
#endif // IBM

#if defined(IBM) || defined(VISCOELASTIC)  // Only pure IBM does not need UPDATE_RHO & UPDATE_U
	#define UPDATE_RHO // density field need to be updated exactly every LBM step
	#define UPDATE_U // velocity field need to be updated exactly every LBM step
#endif // IBM || VISCOELASTIC

#ifdef TETRA
	#undef MEMBRANE_FORCES
#endif // TETRA



#ifdef MOVING_WALLS
	#undef EQUILIBRIUM_BOUNDARIES
#endif // MOVING_WALLS
#ifndef THREAD_BLOCK_SIZE
	#define THREAD_BLOCK_SIZE 256 // default 256, best performance for 256, GPU warp size is 32; Actually this probably depends on GPU manufacturer
#endif // THREAD_BLOCK_SIZE

//TODO remove
#ifdef DEBUG  // Not implemented yet
#undef DEBUG
#endif

// Variable data type settings  //TODO move to somewhere sensible

#define IBM_DOUBLE  // Calculate ibm in double precission, otherwise use float
#define LBM_DOUBLE  // Calculate lbm in double precission, otherwise use float
#define VISCOELASTIC_DOUBLE  // Calculate viscoelastic force in double precission, otherwise use float
#define DDF_DOUBLE  // Store desity distribution function in double precission, otherwise ise float
#define FORCE_DOUBLE  // Calculate and accumulate forces in double precission; Currently affects: IBM

#ifdef FORCE_DOUBLE
	#define IBM_DOUBLE
	#define TETRA_DOUBLE
#endif

#ifdef FORCE_DOUBLE
	typedef double forcePrecisionFloat;
#else  // !FORCE_DOUBLE
	typedef float forcePrecisionFloat;
#endif  // FORCE_DOUBLE

#ifdef IBM_DOUBLE
typedef double ibmPrecisionFloat;
//typedef double3 ibmPrecisionFloat3;
#else  // !IBM_DOUBLE
typedef float ibmPrecisionFloat;
//typedef float3 ibmPrecisionFloat3;
#endif  // IBM_DOUBLE

#ifdef TETRA_DOUBLE
typedef double tetraPrecisionFloat;
//typedef double3 tetraPrecisionFloat3;
#else  // !TETRA_DOUBLE
typedef float tetraPrecisionFloat;
//typedef float3 tetraPrecisionFloat3;
#endif  // TETRA_DOUBLE

#ifdef LBM_DOUBLE
typedef double lbmPrecisionFloat;
//typedef double3 lbmPrecisionFloat3;
#else  // !LBM_DOUBLE
typedef float lbmPrecisionFloat;
//typedef float3 lbmPrecisionFloat3;
#endif  // LBM_DOUBLE

#ifdef VISCOELASTIC_DOUBLE
typedef double viscoelasticPrecisionFloat;
//typedef double3 viscoelasticPrecisionFloat3;
#else  // !LBM_VISCOELASTIC
typedef float viscoelasticPrecisionFloat;
//typedef float3 viscoelasticPrecisionFloat3;
#endif  // LBM_VISCOELASTIC

#ifdef DDF_DOUBLE
typedef double ddfPrecisionFloat;
#else  // !DDF_DOUBLE
typedef float ddfPrecisionFloat;
#endif  // DDF_DOUBLE
