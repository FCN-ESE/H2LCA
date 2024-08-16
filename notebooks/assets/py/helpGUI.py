'''

This function is the function which the GUI adresses.

The main function is the runGUI-function.

'''

from __future__ import division
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import time
import datetime
import copy
import assets.py.plotFunctions as pF
# import plotFunctions as pF
import pyomo.environ as pyomo
import pyomo.opt as opt
from contextlib import contextmanager, redirect_stdout
# import pandas as pd

@contextmanager
def suppress_stdout():
	with open(os.devnull, "w") as devnull:
		old_stdout = sys.stdout
		sys.stdout = devnull
		try:
			yield
		finally:
			sys.stdout = old_stdout

def RBF(interest_rate_input,T):
        
    if interest_rate_input == 0.0:
        Rentenbarwertfaktor = T
    else:
        Rentenbarwertfaktor = ((1+interest_rate_input)**T-1)/((1+interest_rate_input)**T*interest_rate_input)
    
    return Rentenbarwertfaktor

def runGUI(language_strings, load_t, wind_cap_credits, pv_cap_credits, ghg_cost, ghg_max, ghg_max_act, no_nuclear, no_lignite, no_black_coal, renewables, interest_rate, min_cap_renewables,
		   n_steps, gas_props, cc_gas_props, lignite_props, black_coal_props, nuclear_props, wind_props, pv_props):
	'''
	main function which does all the PREPROCESSING, OPTIMIZATION, and POSTPROCESSING and takes the following arguments:

		:parameter language_strings: xml-tree object of strings
		:parameter ghg_cost: integer value for costs for CO2-emissions in €/t
		:parameter ghg_max: integer value for maximum value for the CO2-emissions in % (referencing to 1000Mt)
		:parameter ghg_max_act: boolean value if there exists a maximum value for the CO2-emissions
		:parameter no_nuclear: boolean value if nuclear power is forbidden (True -> Nuclear forbidden)
		:parameter no_lignite: boolean value if lignite power is forbidden (True -> Lignite forbidden)
		:parameter no_black_coal: boolean value if black coal power is forbidden (True -> Black Coal forbidden)
		:parameter interest_rate: float value for the interest rate in %
		:parameter renewables: boolean value if there exist renewable energies on the market
		:parameter min_cap_renewables: boolean value if there exist a minimum of 10GW for renewable energies
		:parameter n_steps: integer value number of time steps (10 or 8760 time steps per year)
		:parameter gas_props, cc_gas_props, lignite_props, ..., pv_props: np-array for properties of different technologies
				   xxProps[0]: integer value for investment costs in €/kW
				   xxProps[1]: integer value for lifetime in years
				   xxProps[2]: float value for fuel costs in €/MWh_therm (=0 for Wind & PV)
				   xxProps[3]: float value for operating costs in €/MWh_el (=0 for Wind & PV)
				   xxProps[4]: integer value for efficiency in % (value for Wind & PV by renewables share .csv)
				   xxProps[5]: float value for specific CO2-emissions in kg/MWh_therm (=0 for Nuclear, Wind & PV)

		:return results_cap: list of length 7, float value for installed capacity for each technology in MW
				results_pow: list of length 7, float value for sum of dispatched powers for each technology in W
				results_ghg: list of length 7, float value for CO2-emissions for each technology in t
				case_id: string of the case ID for the simulated output (path to output)
				target_value: float, objective value

	'''

	# preparing folder for all files related to this optimization case
	ts = time.time()
	ts_readable = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
	case_id = "output/scenario_game_" + ts_readable
	if not os.path.exists(case_id):
		os.makedirs(case_id)
	if not os.path.exists(case_id +'/charts'):
		os.makedirs(case_id +'/charts')
	case_id = case_id + "/"


	''' 
	1. Prepare the input data for the LP model
	'''
	
	# introduce sets of indices
	T = list(range(n_steps)) # set of timesteps
	G = ['CCGas','Gas','BlackCoal','Lignite','Nuclear','Wind','PV'] # generator technologies

	# constant length of timesteps
	LENGTH = 8760/n_steps

	interest_rate = interest_rate / 100

	# convention:   tech_data[(g,'invest')] = investment costs of technology g [€/kW_el]
	#               tech_data[(g,'eta_el')] = efficiency factor of technology g [1]
	#               tech_data[(g,'fuel price')]= fuel price of technology g [€/MWh_thermal]
	#               tech_data[(g,'other variable costs')] = other variable costs of technology g [€/MWh_el]
	#               tech_data[(g,'emission factor')] = emission factor of technology g [kg CO2/MWh_thermal]
	tech_data_g = {('CCGas','invest'):  cc_gas_props[0],		('CCGas','RBF'):   	RBF(interest_rate,cc_gas_props[1]),		('CCGas','eta_el'):   	cc_gas_props[4]/100,	('CCGas','fuel price'):   	cc_gas_props[2],	('CCGas','other variable costs'):    	cc_gas_props[3],		('CCGas','emissions'):   	cc_gas_props[5], 
				('Gas','invest'):      	gas_props[0],			('Gas','RBF'):      RBF(interest_rate,gas_props[1]),		('Gas','eta_el'):     	gas_props[4]/100,		('Gas','fuel price'):     	gas_props[2],		('Gas','other variable costs'):      	gas_props[3],			('Gas','emissions'):     	gas_props[5], 
				('BlackCoal','invest'): black_coal_props[0],	('BlackCoal','RBF'):RBF(interest_rate,black_coal_props[1]),	('BlackCoal','eta_el'): black_coal_props[4]/100,('BlackCoal','fuel price'): black_coal_props[2],('BlackCoal','other variable costs'):   black_coal_props[3],	('BlackCoal','emissions'):  black_coal_props[5],
				('Lignite','invest'):	lignite_props[0],		('Lignite','RBF'):	RBF(interest_rate,lignite_props[1]),	('Lignite','eta_el'):	lignite_props[4]/100,	('Lignite','fuel price'): 	lignite_props[2],	('Lignite','other variable costs'):		lignite_props[3],		('Lignite','emissions'):	lignite_props[5], 
				('Nuclear','invest'):	nuclear_props[0],		('Nuclear','RBF'):	RBF(interest_rate,nuclear_props[1]),	('Nuclear','eta_el'):	nuclear_props[4]/100,	('Nuclear','fuel price'): 	nuclear_props[2],	('Nuclear','other variable costs'): 	nuclear_props[3],		('Nuclear','emissions'):	nuclear_props[5], 
				('Wind','invest'):		wind_props[0],			('Wind','RBF'):		RBF(interest_rate,wind_props[1]),		('Wind','eta_el'):		wind_props[4]/100,		('Wind','fuel price'): 		wind_props[2],		('Wind','other variable costs'): 		wind_props[3],			('Wind','emissions'):  		wind_props[5], 
				('PV','invest'):		pv_props[0],			('PV','RBF'):		RBF(interest_rate,pv_props[1]),			('PV','eta_el'):		pv_props[4]/100,		('PV','fuel price'): 		pv_props[2],		('PV','other variable costs'): 			pv_props[3],			('PV','emissions'):  		pv_props[5]}
							
	# peak load [MW]
	peak_load = 79487

	
	'''
	2. Build the LP model
	'''

	model = pyomo.ConcreteModel()

	# define sets of the LP model  
	model.T = pyomo.Set(initialize=T)
	model.G = pyomo.Set(initialize=G)

	# create decision variables
	model.x_g_t = pyomo.Var(model.G, model.T, domain=pyomo.NonNegativeReals)    # dispatched power of technology g during hour t [MW]
	model.y_g = pyomo.Var(model.G, domain=pyomo.NonNegativeReals)               # installed capacity per technology g [MW]
	
	# add constraints
	
	# cover demand
	def define_demand_restriction(model, t):
		return sum(model.x_g_t[g, t] for g in model.G) == load_t[t]
	model.demand_restriction = pyomo.Constraint(model.T, rule=define_demand_restriction)

	# restrict ghg emissions
	if ghg_max_act:
		# print(sum(tech_data_g[('Gas','emissions')] / tech_data_g[('Gas','eta_el')] * load_t[i] * LENGTH for i in range(10)))
		# print(ghg_max * 1000 * 1e6 * 1e3)
		def define_ghg_restriction(model):
			return sum((tech_data_g[(g,'emissions')] / tech_data_g[(g,'eta_el')]) * model.x_g_t[g, t] * LENGTH for g in model.G for t in model.T) <= (ghg_max/100) * 1000 * 1e6 * 1e3 # ghg_max in % of 1000 Mt
		model.ghg_restriction = pyomo.Constraint(rule=define_ghg_restriction)

	# capacity restriciton and variable connection
	def define_capacity_restriciton(model, g, t):
		return model.x_g_t[g,t] <= model.y_g[g]
	model.capacity_restriciton = pyomo.Constraint(model.G, model.T, rule=define_capacity_restriciton)

	def define_peak_load_restriciton(model):
		return sum(model.y_g[g] for g in model.G) >= peak_load	
	model.peak_load_restriciton = pyomo.Constraint(rule=define_peak_load_restriciton)

	def define_wind_generation_restriction(model, t):
		return model.x_g_t['Wind',t] <= wind_cap_credits[t] * model.y_g['Wind']
	model.wind_generation_restriction = pyomo.Constraint(model.T, rule=define_wind_generation_restriction)

	def define_pv_generation_restriction(model, t):
		return model.x_g_t['PV',t] <= pv_cap_credits[t] * model.y_g['PV']
	model.pv_generation_restriction = pyomo.Constraint(model.T, rule=define_pv_generation_restriction)

	if no_nuclear:
		def define_no_nuclear_restriciton(model):
			return model.y_g['Nuclear'] == 0.0
		model.no_nuclear_restriciton = pyomo.Constraint(rule=define_no_nuclear_restriciton)
	if no_lignite:
		def define_no_lignite_restriciton(model):
			return model.y_g['Lignite'] == 0.0
		model.no_lignite_restriciton = pyomo.Constraint(rule=define_no_lignite_restriciton)
	if no_black_coal:
		def define_no_black_coal_restriciton(model):
			return model.y_g['BlackCoal'] == 0.0
		model.no_black_coal_restriciton = pyomo.Constraint(rule=define_no_black_coal_restriciton)
	if not renewables:
		def define_no_wind_restriciton(model):
			return model.y_g['Wind'] == 0.0
		model.no_wind_restriciton = pyomo.Constraint(rule=define_no_wind_restriciton)
		def define_no_pv_restriciton(model):
			return model.y_g['PV'] == 0.0
		model.no_pv_restriciton = pyomo.Constraint(rule=define_no_pv_restriciton)
	elif min_cap_renewables:
		def define_min_renewables_restriciton(model):
			return model.y_g['Wind'] + model.y_g['PV'] >= 10000.0
		model.min_renewables_restriciton = pyomo.Constraint(rule=define_min_renewables_restriciton)
		

	# define objective function (i.e. dispatch costs)
	def define_objective_function(model):
		return sum(tech_data_g[(g,'invest')] * 1000 / tech_data_g[(g,'RBF')] * model.y_g[g] for g in model.G) + \
			sum(((tech_data_g[(g,'fuel price')] + tech_data_g[(g,'emissions')] * (ghg_cost/1000)) / tech_data_g[(g,'eta_el')] + \
					tech_data_g[(g,'other variable costs')]) * model.x_g_t[g,t] * LENGTH for g in model.G for t in model.T)
	model.Obj = pyomo.Objective(rule=define_objective_function, sense=pyomo.minimize)

	# write problem to readable lp-file --> good for debugging
	model.write(case_id + 'scenario_game_lp_formulation.lp', io_options={'symbolic_solver_labels':True})


	''' 
	ACTUAL OPTIMIZATION
	'''

	# call solver
	model.dual = pyomo.Suffix(direction=pyomo.Suffix.IMPORT_EXPORT)
	model.junk = pyomo.Suffix()
	optimizer = opt.SolverFactory('glpk')
	solved_model = optimizer.solve(model, tee=True)

	'''
	POSTPROCESSING
	'''

	# get demand from consumer
	demand = load_t

	# get installed Capacity Data
	gas_cap = pyomo.value(model.y_g["Gas"])
	cc_gas_cap = pyomo.value(model.y_g["CCGas"])
	lignite_cap = pyomo.value(model.y_g["Lignite"])
	black_coal_cap = pyomo.value(model.y_g["BlackCoal"])
	nuclear_cap = pyomo.value(model.y_g["Nuclear"])
	wind_cap = pyomo.value(model.y_g["Wind"])
	pv_cap = pyomo.value(model.y_g["PV"])

	# get dispatched Power Data
	gas_disp_pow = [pyomo.value(model.x_g_t["Gas", t]) for t in model.T]
	cc_gas_disp_pow = [pyomo.value(model.x_g_t["CCGas", t]) for t in model.T]
	lignite_disp_pow = [pyomo.value(model.x_g_t["Lignite", t]) for t in model.T]
	black_coal_disp_pow = [pyomo.value(model.x_g_t["BlackCoal", t]) for t in model.T]
	nuclear_disp_pow = [pyomo.value(model.x_g_t["Nuclear", t]) for t in model.T]
	wind_disp_pow = [pyomo.value(model.x_g_t["Wind", t]) for t in model.T]
	pv_disp_pow = [pyomo.value(model.x_g_t["PV", t]) for t in model.T]

	# get CO2-emissions
	gas_ghg = sum(gas_disp_pow) * LENGTH * (tech_data_g[('Gas','emissions')] / tech_data_g[('Gas','eta_el')])
	cc_gas_ghg = sum(cc_gas_disp_pow) * LENGTH * (tech_data_g[('CCGas','emissions')] / tech_data_g[('CCGas','eta_el')])
	lignite_ghg = sum(lignite_disp_pow) * LENGTH * (tech_data_g[('Lignite','emissions')] / tech_data_g[('Lignite','eta_el')])
	black_coal_ghg = sum(black_coal_disp_pow) * LENGTH * (tech_data_g[('BlackCoal','emissions')] / tech_data_g[('BlackCoal','eta_el')])
	all_ghg = gas_ghg + cc_gas_ghg + lignite_ghg + black_coal_ghg


	# The parameter of the getDualsolLinear should be constriants
	'''
	This function is used to calculate the shadow price of electricity by getting
	the dual variables of the equality constraints( dispatched + import = demand + export)
	These equality constraints are setted at each time step, so by doing so we can
	get the eletricity price at each time step.
	
	In the toScip.py file, the equaCons is returned to save these above constriants.
	Here, we want to get the dual variables of these constriants.
	'''
		
	elec_prices_shadow = [model.dual[model.demand_restriction[t]]/LENGTH for t in model.T]

	#print("Printing Shadow price to file.")
	fileName = case_id + "elecShadowPrices.txt"
	with open(fileName, 'w') as f:
		for item in elec_prices_shadow:
			f.write("%s\n" % item)
		f.close()
		
	elec_prices_margin = pF.calcElecPrices(gas_disp_pow, cc_gas_disp_pow, lignite_disp_pow, black_coal_disp_pow, nuclear_disp_pow, wind_disp_pow, pv_disp_pow, n_steps, ghg_cost, gas_props, cc_gas_props, lignite_props, black_coal_props, nuclear_props, wind_props, pv_props)
	#print("Printing Margin price to file.")
	fileName = case_id + "elecMarginPrices.txt"
	with open(fileName, 'w') as f:
		# for item in elec_pricesMargin:
		for item in elec_prices_margin:
			f.write("%s\n" % item)
		f.close()


	'''
	PLOT RESULT CHARTS
	'''

	# Pie plot for capacity shares for the different technologies
	pF.piePlot(gas_cap, cc_gas_cap, lignite_cap, black_coal_cap, nuclear_cap, wind_cap, pv_cap, case_id, language_strings, 'Cap')

	# Pie plot for CO2-emissions for the different technologies (only created if CO2-emissions are allowed)
	if ghg_max == 0 and ghg_max_act==True:
		pass
	else:
		pF.piePlot(gas_ghg, cc_gas_ghg, lignite_ghg, black_coal_ghg, 0, 0, 0, case_id, language_strings, 'CO2')

	# chart for the supplied power at each simulated time step
	pF.supplyProfileChart(gas_disp_pow, cc_gas_disp_pow, lignite_disp_pow, black_coal_disp_pow, nuclear_disp_pow,
						  wind_disp_pow, pv_disp_pow, case_id, 1, n_steps, language_strings)

	# chart for the demand at each simulated time step
	pF.demandChart(demand, n_steps, 1, case_id, language_strings)

	# chart for electricity-prices at each simulated time step
	pF.priceChart(elec_prices_margin, n_steps, 1, case_id, language_strings)

	# bar chart to show dispatched energy in each time step (only for 10 time steps)
	pF.barChart(gas_disp_pow, cc_gas_disp_pow, lignite_disp_pow, black_coal_disp_pow, nuclear_disp_pow, wind_disp_pow,
				pv_disp_pow, case_id, 1, n_steps, language_strings)


	'''
	SAVE DATA FOR RETURN
	'''

	results_cap = [gas_cap, cc_gas_cap, lignite_cap, black_coal_cap, nuclear_cap, wind_cap, pv_cap]

	results_dis_power = [gas_disp_pow, cc_gas_disp_pow, lignite_disp_pow, black_coal_disp_pow,
				  	   nuclear_disp_pow, wind_disp_pow, pv_disp_pow]

	results_ghg = [all_ghg, gas_ghg, cc_gas_ghg, lignite_ghg, black_coal_ghg]

	target_value = round(model.Obj.expr(),2)
	


	return results_cap, results_dis_power, results_ghg, case_id, target_value, elec_prices_margin

if __name__ == "__main__":
	runGUI(language_strings = None,
                              load_t = [73409,69648,66632,63454,59526,55367,51970,49045,46030,41466],
                              pv_cap_credits = [0.1 for i in range(10)],
                              wind_cap_credits = [0.1 for i in range(10)],
                              ghg_cost = 0, 
                              ghg_max = 20, 
                              ghg_max_act = True, 
                              no_nuclear = True, 
                              no_lignite = True, 
                              no_black_coal = True, 
                              renewables = True, 
                              interest_rate = 0.08, 
                              min_cap_renewables = 10,
                              n_steps = 10, 
                              gas_props = [450,20,28.0,3.6,37,204.8],
							  cc_gas_props = [500,20,28.0,5.23,55,204.8],
							  lignite_props = [1400,30,7.5,13.44,39,400],
							  black_coal_props = [1300,30,10.47,9.7,42,342],
							  nuclear_props = [2700,40,3.47,7.46,35,0.0],
							  wind_props = [1000,20,0.0,0.0,1.0,0.0],
							  pv_props = [1400,20,0.0,0.0,1.0,0.0],
							  )