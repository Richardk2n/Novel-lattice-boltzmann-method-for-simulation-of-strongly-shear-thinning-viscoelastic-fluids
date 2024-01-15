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
#include <numbers>
using std::numbers::pi;

#include <fmt/core.h>
// using fmt::format
using fmt::print;
#include <fmt/color.h>
using fmt::color;

#include <spdlog/spdlog.h>
//using spdlog::info;

#include "fieldWriter.hpp"
using fluidx3d::io::fieldWriter::FieldWriter;
#include "io.hpp"
using fluidx3d::util::io::IO;
#include "poiseuilleFlow.hpp"
using fluidx3d::util::poiseuilleFlow::CircularPoiseuilleFlow;
#include "stringTools.hpp"
using fluidx3d::util::stringTools::split;

void example(SimulationOptions& options, Parameters& parameters) {
	json raw = parameters.getRaw();
	const int id = raw.at("id");
	//const string path_files = "/tp6-gekle/nas/bt307867/cellsim/novel_2023-10-24/" + std::to_string(id) + "/";
	const string path_files = fmt::format("/tp6-gekle/nas/bt307867/cellsim/novel_2024-01-15/{}/", id); // id = 1
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
	uint Ly = 43;
	uint Lz = 43;

	/*
	 * this parameter decreases the viscosity
	 * simultaneously k1 and kb are increased to keep Ca and kbHat constant
	 * --> leads to larger time step (good)
	 * --> and larger Ma (not good, but acceptable)
	 */
	const double ReScale = 1e0;

	const long numberSteps = 2'000'000'000;//(long)(10000000*25/dtScale/ReScale);
	const uint vtkInterval = 500'000;//(uint)(500*5*25/dtScale/ReScale);  // every vtkInterval steps a vtk file is written
	initInfo(numberSteps);

	const double R_SI = 10e-6;
	const double R = 20.5;
	const double grid_SI = R_SI/R; // Cell has a radius of 15 Lattice units
	const double rho_SI = 1e3f;


	// viscosity of alginate
	const double viscosityShuffleFraction = raw.at("viscosityShuffleFraction");
	const double eta_solvent_SI = 1e-3/ReScale;
	const double eta_polymer_SI = 48.2/ReScale;
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
	const double p0 = rho0*V0*nu0/L0;
	spdlog::info("p0: {}", p0);
	spdlog::info("V0: {}", V0);
	const double shearRateSI = 4;
	const double Vmax_SI = shearRateSI*R_SI;

	units.set_m_kg_s(1.0f, 1, rho, grid_SI, V0, rho_SI);


	const double lambda_polymer_SI = 0.343;
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
	options.ptt_epsilon.set(0.545);
	options.ptt_xi.set(0);

	spdlog::info("T0: {}", T0);
	const double Re = Vmax_SI/V0*R/(nu*eta_solvent_SI/mu_SI);
	spdlog::info("Re = {}", Re);

	const double px = (double)raw.at("pxSI")/(p0/L0); // This as a approximation for small eta_s

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

	vector<int> ys;
	vector<int> zs;
	vector<double> Ns;
	vector<double> VSs;

	string lines = IO::readFile("preset3D.csv");
	for(string line : split(lines, "\n")) {
		if (line == "") {
			continue;
		}
		auto parts = split(line, ";");
		ys.push_back(stoi(parts[0]));
		zs.push_back(stoi(parts[1]));
		Ns.push_back(stod(parts[2]));
		VSs.push_back(stod(parts[3]));
	}

	// Define geometry
	for(uint n=0, x=0, y=0, z=0, sx=lattice.size_x(), sy=lattice.size_y(), sz=lattice.size_z(), s=sx*sy*sz; n<s; n++, x=n%(s/sz)%sx, y=n%(s/sz)/sx, z=n/sx/sy) {
		int ry = y - (Ly-1)/2;
		int rz = z - (Lz-1)/2;
		double r = std::hypot(ry, rz);
		if(r >= (Ly-2)/2.){
			lattice.flags[n] = TYPE_W;
		}else{
			lattice.flags[n] = TYPE_F; //TODO should be globally set
			for(int i = 0; i < ys.size(); i++) { // Yes, this is terrible
				if(std::abs(ry) == ys[i] && std::abs(rz) == zs[i]) {
					double xy = 0, xz = 0;
					if(ry == 0){
						xz = (rz >= 0? 1: -1)*VSs[i]/(eta_polymer_SI/lambda_polymer_SI);
					} else if(rz == 0) {
						xy = (ry >= 0? 1: -1)*VSs[i]/(eta_polymer_SI/lambda_polymer_SI);
					} else {
						double angle = std::atan(rz/(double)ry);
						xy = (ry>=0?1:-1)*std::cos(angle)*VSs[i]/(eta_polymer_SI/lambda_polymer_SI);
						xz = (ry>=0?1:-1)*std::sin(angle)*VSs[i]/(eta_polymer_SI/lambda_polymer_SI);
					}
					lattice.polymerConformationTensor[n] = Ns[i]/(eta_polymer_SI/lambda_polymer_SI);
					lattice.polymerConformationTensor[3*s+n] = xy;
					lattice.polymerConformationTensor[5*s+n] = xz;
				}
			}
		}
	}
	lattice.polymerConformationTensor.write_to_gpu(1);

	spdlog::info("Going to run {:e} steps...", (double)numberSteps);
	spdlog::info("Which is equivalent to {:e} seconds.", numberSteps*T0);

	run(0);

	FieldWriter fw_flags(lattice.flags, {Lx, Ly, Lz});
	fw_flags.write<uchar>(path_vtk+"flags_0.vtk");

	FieldWriter fw_u(lattice.u, {Lx, Ly, Lz});
	FieldWriter fw_strainRateTensor(lattice.u, {Lx, Ly, Lz});
	FieldWriter fw_viscousStressTensor(lattice.u, {Lx, Ly, Lz});
	FieldWriter fw_polymer_conformationTensor(lattice.polymerConformationTensor, {Lx, Ly, Lz});

	// function to write vtk files
	auto vtk_routine = [&]() {
		const string step_string_vtk = fmt::format("_{}.vtk", lattice.get_time_step());
		fw_u.write<float>(fmt::format("{}u{}", path_vtk_velocity, step_string_vtk));
		fw_strainRateTensor.write<float>(fmt::format("{}S{}", path_vtk_strainRateTensor, step_string_vtk));
		fw_viscousStressTensor.write<float>(fmt::format("{}VS{}", path_vtk+"viscous_stress/", step_string_vtk));
		fw_polymer_conformationTensor.write<float>(fmt::format("{}CT{}", path_vtk+"polymer_conformationTensor/", step_string_vtk));
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
