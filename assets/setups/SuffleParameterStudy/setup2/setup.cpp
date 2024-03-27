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
#include <iostream>
using std::endl;

#include <fmt/format.h>
// using fmt::format
using fmt::print;
#include <fmt/color.h>
using fmt::color;

#include <spdlog/spdlog.h>
//using spdlog::info;

#include "cellWriter.hpp"
using fluidx3d::io::cellWriter::CellWriter;
#include "fieldWriter.hpp"
using fluidx3d::io::fieldWriter::FieldWriter;
#include "io.hpp"
using fluidx3d::util::io::IO;
#include "poiseuilleFlow.hpp"
using fluidx3d::util::poiseuilleFlow::CircularPoiseuilleFlow;

void example(SimulationOptions& options, Parameters& parameters) {
	json raw = parameters.getRaw();
	const int id = raw.at("id");
	//const string path_files = "/tp6-gekle/nas/bt307867/cellsim/novel_2023-10-24/" + std::to_string(id) + "/";
	const string path_files = fmt::format("SuffleParameterStudy/{}/", id);
	const path filesPath(path_files);
	const path vtkPath = filesPath/"vtkfiles";
	const path path_vtk_velocity = vtkPath/"velocity";
	const path path_vtk_strainRateTensor = vtkPath/"strainRateTensor";
	const path path_vtk_viscousStressTensor = vtkPath/"viscousStressTensor";
	const path path_vtk_polymerConformationTensor = vtkPath/"polymerConformationTensor";

	/*
	 * this parameter increases the time step
	 * --> leads to larger relaxation time tau (not good, but acceptable)
	 * --> leads to larger Ma (not good, but acceptable)
	 * --> eventually may lead to inaccuracies
	 */
	const double dtScale = 1e0;  // decrease timestep

	// Length of simulation box
	uint Lx = 2;
	uint Ly = 43;
	uint Lz = 2;

	/*
	 * this parameter decreases the viscosity
	 * simultaneously k1 and kb are increased to keep Ca and kbHat constant
	 * --> leads to larger time step (good)
	 * --> and larger Ma (not good, but acceptable)
	 */
	const double ReScale = 1e0;

	const long numberSteps = 10'000'000;//(long)(10000000*25/dtScale/ReScale);
	const uint vtkInterval = 100'000;//(uint)(500*5*25/dtScale/ReScale);  // every vtkInterval steps a vtk file is written
	initInfo(numberSteps);

	const double R_SI = 10e-6;
	const double R = 20.5;
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
	const double T0 = pow(L0, 2)/nu0;
	const double V0 = L0 / T0;
	const double p0 = rho0*V0*V0;
	spdlog::info("p0: {}", p0);
	spdlog::info("V0: {}", V0);
	const double shearRateSI = raw.at("shearRateSI");
	const double Vmax_SI = shearRateSI*R_SI;
	const double Vmax = Vmax_SI/V0;

	units.set_m_kg_s(1.0, 1, rho, grid_SI, V0, rho_SI);


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
	const double Re = Vmax_SI/V0*R/(nu*eta_solvent_SI/mu_SI);
	spdlog::info("Re = {}", Re);

	//									force in x		y	z	direction;
	set_common_options(options, Lx, Ly, Lz, nu, 0.0, 0.0, 0.0);

	initialize(options);

	std::stringstream params;
	params << "\nT = " << numberSteps << "\nnu = " << nu << "\nrho = " << rho << "\nR = " << R << "\n";
	params << "\nT_SI = " << numberSteps*T0 << "\ndt_SI = " << T0 <<  "\nnu_SI = " << nu_SI << "\nrho_SI = " << rho_SI << "\nR_SI = " << R_SI << "\n";
	params << "\nLx = " << Lx << "\nLy = " << Ly <<  "\nLz = " << Lz;
	params << "\nnu = " << nu;
	IO::writeFile(path_files+"params.dat", params.str()); // TODO replace with json

	json& info = parameters.getInfo();
	info["conversions"] = {{"L0", L0}, {"T0", T0}, {"V0", V0}};


	// Define geometry
	for(uint n=0, x=0, y=0, z=0, sx=lattice.size_x(), sy=lattice.size_y(), sz=lattice.size_z(), s=sx*sy*sz; n<s; n++, x=n%(s/sz)%sx, y=n%(s/sz)/sx, z=n/sx/sy) {
		if(y == 0 || y == Ly-1){
			lattice.flags[n] = TYPE_VW;
			lattice.u[n] = (y==0?-1:1)*Vmax;
		}else{
			lattice.flags[n] = TYPE_F; //TODO should be globally set  //TODO does not seem to be an issue??? Or is this messing with my sims?
		}
	}

	spdlog::info("Going to run {:e} steps...", (double)numberSteps);
	spdlog::info("Which is equivalent to {:e} seconds.", numberSteps*T0);

	run(0);

	FieldWriter fw_flags(lattice.flags, {Lx, Ly, Lz});
	fw_flags.write<uchar>(vtkPath/"flags_0.vtk");

	FieldWriter fw_u(lattice.u, {Lx, Ly, Lz});
	FieldWriter fw_strainRateTensor(lattice.strainRateTensor, {Lx, Ly, Lz});
	FieldWriter fw_viscousStressTensor(lattice.viscousStressTensor, {Lx, Ly, Lz});
	FieldWriter fw_polymer_conformationTensor(lattice.polymerConformationTensor, {Lx, Ly, Lz});

	// function to write vtk files
	auto vtk_routine = [&]() {
		const long step = lattice.get_time_step();
		fw_u.write<float>(path_vtk_velocity/fmt::format("u_{}.vtk", step));
		fw_strainRateTensor.write<float>(path_vtk_strainRateTensor/fmt::format("D_{}.vtk", step));
		fw_viscousStressTensor.write<float>(path_vtk_viscousStressTensor/fmt::format("S_{}.vtk", step));
		fw_polymer_conformationTensor.write<double>(path_vtk_polymerConformationTensor/fmt::format("A_{}.vtk", step));
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
