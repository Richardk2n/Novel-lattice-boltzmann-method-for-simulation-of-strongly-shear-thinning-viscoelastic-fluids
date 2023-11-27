#include "setup.hpp"
#include "shapes.hpp"
#include "opencl.hpp" // contains "options.hpp", "main.hpp"
#include "ibm.hpp"
#include <numeric>
#include <sstream>

#include <filesystem>
using std::filesystem::path;
#include <fstream>
using std::ofstream;

#include <fmt/core.h>
// using fmt::format
using fmt::print;
#include <fmt/color.h>
using fmt::color;

#include <spdlog/spdlog.h>
//using spdlog::info;

#include "io.hpp"
using fluidx3d::util::io::IO;
#include "poiseuilleFlow.hpp"
using fluidx3d::util::poiseuilleFlow::CircularPoiseuilleFlow;

void example(SimulationOptions& options, Parameters& parameters) {
	json raw = parameters.getRaw();
	const int id = raw.at("id");
	//const string path_files = "/tp6-gekle/nas/bt307867/cellsim/novel_2023-10-24/" + std::to_string(id) + "/";
	const string path_files = fmt::format("Nozzle/{}/", id);
	const path filesPath(path_files);
	const string path_vtk = path_files + "vtkfiles/";
	const string path_vtk_velocity = path_vtk + "velocity/";
	const string path_vtk_strainRateTensor = path_vtk + "strainRateTensor/";
	const string path_vtk_viscousStressTensor = path_vtk + "viscousStressTensor/";
	const string path_vtk_polymerConformationTensor = path_vtk + "polymerConformationTensor/";

	/*
	 * this parameter increases the time step
	 * --> leads to larger relaxation time tau (not good, but acceptable)
	 * --> leads to larger Ma (not good, but acceptable)
	 * --> eventually may lead to inaccuracies
	 */
	const double dtScale = 1e0;  // decrease timestep

	// Length of simulation box
	uint Lx = 2;
	uint Ly = 191;
	uint Lz = 191;

	/*
	 * this parameter decreases the viscosity
	 * simultaneously k1 and kb are increased to keep Ca and kbHat constant
	 * --> leads to larger time step (good)
	 * --> and larger Ma (not good, but acceptable)
	 */
	const double ReScale = 1e0;

	const long numberSteps = 5'000'000;//(long)(10000000*25/dtScale/ReScale);
	const uint vtkInterval = 50'000;//(uint)(500*5*25/dtScale/ReScale);  // every vtkInterval steps a vtk file is written
	initInfo(numberSteps);

	const double R_SI = 1.8e-3;
	const double R = 94.5;
	const double grid_SI = R_SI/R; // Cell has a radius of 15 Lattice units
	const double rho_SI = 1e3f;


	// viscosity of alginate
	const double viscosityShuffleFraction = raw.at("viscosityShuffleFraction");
	const double eta_solvent_SI = 1e-3/ReScale;
	const double eta_polymer_SI = 18.7e-3/ReScale;
	const double mu_SI = eta_solvent_SI + viscosityShuffleFraction*eta_polymer_SI;

	const double nu_SI = mu_SI / rho_SI;

	// fix three scales
	const double rho = 1.0f;
	const double nu = 1.0f/6.0f * dtScale;  //TODO Not sure if correct. Should probably ask someone.
	const double L0 = grid_SI;
	const double rho0 = rho_SI/rho;
	const double nu0 = nu_SI/nu;
	const double mu0 = nu0*rho0;

	// determine other scales
	const double E0 = rho0 * sq(nu0) * L0;
	spdlog::info("E0: {}", E0);
	const double T0 = sqrt( rho0 / E0 * pow(L0, 5) );
	const double V0 = L0 / T0;
	const double p0 = rho0*V0*V0;
	spdlog::info("p0: {}", p0);
	spdlog::info("V0: {}", V0);
	const double px_SI = raw.at("px_SI");

	units.set_m_kg_s(1.0f, 1, rho, grid_SI, V0, rho_SI);


	const double lambda_polymer_SI = 0.344e-3;
	const double lambda_polymer = lambda_polymer_SI/T0;
	spdlog::info("lp: {}", lambda_polymer);
	const double eta_polymer = eta_polymer_SI/mu0;
	spdlog::info("ep: {}", eta_polymer);
	spdlog::info("ep/lp: {}", eta_polymer/lambda_polymer);
	spdlog::info("ep/lp_SI: {}", eta_polymer_SI/lambda_polymer_SI);
	//spdlog::info("to_eta: {}", eta_polymer/lambda_polymer*p0*ReScale/shearRateSI);


	options.viscosityShuffleFraction.set(viscosityShuffleFraction);
	options.eta_polymer.set(eta_polymer);
	options.lambda_polymer.set(lambda_polymer);
	options.fv_set.set("D3Q27");
	options.fv_advection_scheme.set("ctu");
	//options.viscoelastic_inout.set("ongoing_shovel");
	options.ptt_epsilon.set(0.27);
	options.ptt_xi.set(0);

	spdlog::info("T0: {}", T0);

	const double px = px_SI/p0*L0;

	//									force in x		y	z	direction;
	set_common_options(options, Lx, Ly, Lz, nu, px, 0.0, 0.0);

	initialize(options);

	std::stringstream params;
	params << "\nT = " << numberSteps << "\nnu = " << nu << "\nrho = " << rho << "\nR = " << R << "\n";
	params << "\nT_SI = " << numberSteps*T0 << "\ndt_SI = " << T0 <<  "\nnu_SI = " << nu_SI << "\nrho_SI = " << rho_SI << "\nR_SI = " << R_SI << "\n";
	params << "\nLx = " << Lx << "\nLy = " << Ly <<  "\nLz = " << Lz;
	params << "\nnu = " << nu;
	IO::writeFile(path_files+"params.dat", params.str()); // TODO replace with json

	json& info = parameters.getInfo();
	info["conversions"] = {{"L0", L0}, {"T0", T0}, {"V0", V0}};
	info["px"] = px;


	// Define geometry
	for(uint n=0, x=0, y=0, z=0, sx=lattice.size_x(), sy=lattice.size_y(), sz=lattice.size_z(), s=sx*sy*sz; n<s; n++, x=n%(s/sz)%sx, y=n%(s/sz)/sx, z=n/sx/sy) {
		if(std::hypot(y - (Ly/2.) + 0.5, z - (Lz/2.) + 0.5) >= R + 0.5) {
			lattice.flags[n] = TYPE_W;
		}else{
			lattice.flags[n] = TYPE_F; //TODO should be globally set  //TODO does not seem to be an issue??? Or is this messing with my sims?
		}
	}

	spdlog::info("Going to run {:e} steps...", (double)numberSteps);
	spdlog::info("Which is equivalent to {:e} seconds.", numberSteps*T0);

	run(0);

	lattice.flags.read_from_gpu();
	lattice.flags.cpu().write_vtk(path_vtk+"flags_0.vtk");

	// function to write vtk files
	auto vtk_routine = [&]() {
		const string step_string_vtk = fmt::format("_{}.vtk", lattice.get_time_step());
		lattice.u.read_from_gpu();
		lattice.u.cpu().write_vtk(fmt::format("{}u{}", path_vtk_velocity, step_string_vtk));
		lattice.strainRateTensor.read_from_gpu();
		lattice.strainRateTensor.cpu().write_vtk(fmt::format("{}S{}", path_vtk_strainRateTensor, step_string_vtk));
		lattice.viscousStressTensor.read_from_gpu();
		lattice.viscousStressTensor.cpu().write_vtk(fmt::format("{}VS{}", path_vtk_viscousStressTensor, step_string_vtk));
		lattice.polymerConformationTensor.read_from_gpu();
		lattice.polymerConformationTensor.cpu().write_vtk(fmt::format("{}C{}", path_vtk_polymerConformationTensor, step_string_vtk));
	};


	vtk_routine();

	parameters.save(filesPath/"parameters.json");

	// Executes an additional full vtkIntervall if numberSteps does not devide neatly
	for(long i = 0; i*vtkInterval < numberSteps; i ++){
		run(vtkInterval);

		vtk_routine();
	}
}

void main_setup(Parameters& parameters) {
	spdlog::info("Loaded parameters: {}", parameters.getRaw().dump());
	SimulationOptions options;
	options.device.set(parameters.getRaw()["gpu"]); // Default gpu
	options.log.set(true); // Likely a good idea??

	example(options, parameters);
}
