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
using std::cout, std::endl; // TODO replace with spdlog

#include <fmt/core.h>
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
using fluidx3d::util::poiseuilleFlow::RectangularPoiseuilleFlow;

void example(SimulationOptions& options, Parameters& parameters) {
	json raw = parameters.getRaw();
	const int id = raw["id"];
	const string path_files = fmt::format("/tp6-gekle/nas/bt307867/cellsim/{}_{}/{}/", "cellInShear", "2024-01-23", id);
	//const string path_files = fmt::format("{}_{}/{}/", "cellInShear", "2024-01-23", id);
	const path filesPath(path_files);
	const string path_vtk = path_files + "vtkfiles/";
	const string path_vtk_cell = path_vtk + "cell/";
	const string path_vtk_inout = path_vtk + "inout/";
	const string path_vtk_velocity = path_vtk + "velocity/";
	const string path_vtk_strainRateTensor = path_vtk + "strainRateTensor/";
	const string path_ibm = "tables/smallTetraCell/";

	/*
	 * this parameter increases the time step
	 * --> leads to larger relaxation time tau (not good, but acceptable)
	 * --> leads to larger Ma (not good, but acceptable)
	 * --> eventually may lead to inaccuracies
	 */
	const float dtScale = 1e0;  // decrease timestep

	// Length of simulation box
	uint Lx = 100;
	uint Ly = 102;
	uint Lz = 100;

	/*
	 * this parameter decreases the viscosity
	 * simultaneously k1 and kb are increased to keep Ca and kbHat constant
	 * --> leads to larger time step (good)
	 * --> and larger Ma (not good, but acceptable)
	 */
	const float ReScale = 1e0;

	const long numberSteps = 40'000'000;//(long)(10000000*25/dtScale/ReScale);
	const uint vtkInterval = 4'000;//(uint)(500*5*25/dtScale/ReScale);  // every vtkInterval steps a vtk file is written
	initInfo(numberSteps);

	const float R_SI = 6e-6;
	const float R = 6.0f;
	const float grid_SI = R_SI/R; // Cell has a radius of 15 Lattice units
	const float rho_SI = 1e3f;


	// viscosity of alginate
	const double viscosityShuffleFraction = 2;
	const float eta_polymer_SI = 18.7e-3/ReScale;
	const float mu_SI = 1e-3/ReScale + viscosityShuffleFraction*eta_polymer_SI;

	double Vmax_SI = 0.02;
	cout << "Vmax_SI: " << Vmax_SI << endl;

	const float nu_SI = mu_SI / rho_SI;

	// fix three scales
	const float rho = 1.0f;
	const float nu = 1.0f/6.0f * dtScale;  //TODO Not sure if correct. Should probably ask someone.
	const float L0 = grid_SI;
	const float rho0 = rho_SI/rho;
	const float nu0 = nu_SI/nu;
	const double mu0 = nu0*rho0;

	tetraPrecisionFloat poissonRatio = 0.48f;//dimless//TODO precission concerns
	tetraPrecisionFloat youngsModulusSI = 1e2; //Pa

	// determine other scales
	const double T0 = pow(L0, 2)/nu0;
	const float V0 = L0 / T0;
	const double p0 = rho0*V0*nu0/L0;
	cout << "p0: " << p0 << endl;
	cout << "V0: " << V0 << endl;

	// convert variables
	const float Vmax = Vmax_SI / V0;
	const tetraPrecisionFloat youngsModulus = youngsModulusSI / p0; // Up for debate

	const double Re_channel = (Ly-2)*grid_SI*Vmax_SI/nu_SI;
	const double Re_cell = R*grid_SI*Vmax_SI/nu_SI;
	const double Ca_tetra = mu_SI*Vmax_SI/(youngsModulusSI*R*L0);
	const double Ma = Vmax*sqrt(3);
	cout << "Re_channel :" << Re_channel << endl;
	cout << "Re_cell :" << Re_cell << endl;
	cout << "Ca_tetra :" << Ca_tetra << endl;
	cout << "Ma :" << Ma << endl;

	units.set_m_kg_s(1.0f, Vmax, rho, grid_SI, Vmax_SI, rho_SI);

	const double lambda_polymer_SI = 0.344e-3;
	const double lambda_polymer = lambda_polymer_SI/T0;
	cout << "lp: " << lambda_polymer << endl;
	const double eta_polymer = eta_polymer_SI/mu0;
	cout << "ep: " << eta_polymer << endl;
	cout << "ep/lp: " << eta_polymer/lambda_polymer << endl;
	cout << "ep/lp_SI: " << eta_polymer_SI/lambda_polymer_SI << endl;
	//cout << "to_eta: " << eta_polymer/lambda_polymer*p0*ReScale/shearRateSI << endl;


	options.viscosityShuffleFraction.set(viscosityShuffleFraction);
	options.eta_polymer.set(eta_polymer);
	options.lambda_polymer.set(lambda_polymer);
	options.fv_set.set("D3Q27");
	options.fv_advection_scheme.set("ctu");
	options.viscoelastic_inout.set("normal_viscoelastic");
	options.ptt_epsilon.set(0.27);
	options.ptt_xi.set(0);

	cout << "T0: " << T0 << endl;

	//									force in x		y		z	direction;
	set_common_options(options, Lx, Ly, Lz, nu, 0.0f , 0.0f, 0.0f);

	// insert cells
	vector<ibmPrecisionFloat3> positions(1);
	{
		positions[0] = ibmPrecisionFloat3(0, 0, 0);//yInit, yInit*5/9.); // wanted offset
	}
	ibm.insert_cells(path_ibm, "softPositions", "softTriangles", "softTetrahedra", 1.0f, positions, 0);
	//ibm.set_inout_contrast(5.0f);  //TODO enable
	//options.ibm_spring_law.set("skalak");


	ibm.getTetra().setPoissonRatio(poissonRatio);
	ibm.getTetra().setYoungsModulus(youngsModulus);

	initialize(options);

	std::stringstream params;
	params << "\nRe = " << R*Vmax / nu << "\nMa = " << Vmax * sqrt(3.0f) << "\n";
	params << "\nT = " << numberSteps << "\nnu = " << nu << "\nrho = " << rho << "\nVmax = " << Vmax << "\nR = " << R << "\n";
	params << "\nT_SI = " << numberSteps*T0 << "\ndt_SI = " << T0 <<  "\nnu_SI = " << nu_SI << "\nrho_SI = " << rho_SI << "\nVmax_SI = " << Vmax_SI << "\nR_SI = " << R_SI << "\n";
	params << "\nLx = " << Lx << "\nLy = " << Ly <<  "\nLz = " << Lz;
	params << "\nnu = " << nu;
	IO::writeFile(path_files+"params.dat", params.str()); // TODO replace with json


	// Define geometry
	for(uint n=0, x=0, y=0, z=0, sx=lattice.size_x(), sy=lattice.size_y(), sz=lattice.size_z(), s=sx*sy*sz; n<s; n++, x=n%(s/sz)%sx, y=n%(s/sz)/sx, z=n/sx/sy) {
		if(y==0 || y==Ly-1){
			lattice.flags[n] = TYPE_W;
			lattice.u[n] = y==0?-Vmax:Vmax;
		}else{
			lattice.flags[n] = TYPE_F; //TODO should be globally set  //TODO does not seem to be an issue??? Or is this messing with my sims?
		}
	}

	spdlog::info("Going to run {} steps...", numberSteps);
	spdlog::info("Which is equivalent to {} seconds.", numberSteps*T0);

	run(0);

	FieldWriter fw_flags(lattice.flags, {Lx, Ly, Lz});
	fw_flags.write<uchar>(path_vtk+"flags_0.vtk");

	std::ofstream maxStretch;
	maxStretch.open(path_files + "maxStretch.dat");
	maxStretch << "timeStep\tstretch_sq\n";

	#ifdef TETRA
		auto& tetras = ibm.getTetra().get_tetras().cpu();
		vector<int> intTetras(tetras.begin(), tetras.end());
		CellWriter cw(ibm.getPoints(), {Lx, Ly, Lz}, {{fluidx3d::io::cellWriter::VTK_TETRA, intTetras}});
	#else
		auto& triangles = ibm.getTriangles().cpu();
		vector<int> intTriangles(triangles.begin(), triangles.end());
		CellWriter cw(ibm.getPoints(), {Lx, Ly, Lz}, {{fluidx3d::io::cellWriter::VTK_TRIANGLE, intTriangles}});
	#endif
	FieldWriter fw_u(lattice.u, {Lx, Ly, Lz});
	FieldWriter fw_strainRateTensor(lattice.strainRateTensor, {Lx, Ly, Lz});
	FieldWriter fw_viscousStressTensor(lattice.viscousStressTensor, {Lx, Ly, Lz});
	FieldWriter fw_inout(lattice.inout, {Lx, Ly, Lz});
	FieldWriter fw_polymer_conformationTensor(lattice.polymerConformationTensor, {Lx, Ly, Lz});

	// function to write vtk files
	auto vtk_routine = [&]() {
		const string step_string_vtk = fmt::format("_{}.vtk", lattice.get_time_step());
		cw.write<float>(fmt::format("{}cells{}", path_vtk_cell, step_string_vtk));
		fw_inout.write<uchar>(fmt::format("{}inout{}", path_vtk_inout, step_string_vtk));
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

		maxStretch << fmt::format("{}\t{}", lattice.get_time_step(), ibm.getMaxStretchSquared()) << endl;
		if(ibm.broken_bonds_check()) {
			spdlog::error("Broken bonds. Aborting this simulation run at step {}.", lattice.get_time_step());
			//break;
		}

	}
	maxStretch.close();
}

void main_setup(Parameters& parameters) {
	spdlog::info("Loaded parameters: {}", parameters.getRaw().dump());
	SimulationOptions options;
	options.device.set(parameters.getRaw()["gpu"]); // Default gpu
	options.log.set(true); // Likely a good idea??

	example(options, parameters);
}
