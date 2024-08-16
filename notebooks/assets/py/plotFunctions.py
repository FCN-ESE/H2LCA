'''

This file contains functions for plotting the scenario game results.

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import figure
import xml.etree.ElementTree as ET


# help functions
def divList(list, x):
    # This function returns a list in which all values are divided by x
    return[i / x for i in list]

def multiList(list, x):
    # This function returns a list in which all values are multiplied by x
    return[i * x for i in list]

def reduceList(list, nSteps):

    i = 0
    newList =[]
    intervall = int(nSteps / 10)

    while i < len(list):
        newList.append(np.mean(list[i:i+intervall+1]))
        i += intervall+1

    return newList

def durationTStep(nSteps):
    # This function returns the duration of one time step in hours
    return 8760/nSteps

def string(name,language_strings):
    # This function returns string from xml-file
    value = language_strings.find("string/[@name='"+name+"']").text
    return value 

# calc functions

def calcKInvCost(Cap, kInv):
    '''
    	:parameter Cap: Installed capacity in MW
    	:parameter kInv: spez. invest cost in €/kW

    	:return the capital costs in €

    '''

    investCost = Cap * kInv * 1000

    return investCost 

def calcRevenue(disp,elecPrice,nSteps):
    '''
    	:parameter disp: dispatched Power in MW
    	:parameter elecPrice: electricity price in €/MWh in each timestep

    	:return the revenue costs in €

    '''
    dispatchedPower = np.array(disp)
    electricityPrice = np.array(elecPrice)

    revenue = dispatchedPower * durationTStep(nSteps) * electricityPrice

    return sum(revenue)

def calcFuelCosts(disp, fuelC, nSteps, efficiency):
    '''
    	:parameter disp: dispatched Power in MW
    	:parameter fuelC: fuel costs in €/MWhtherm -> calculate to €/MWhel with efficiency
    	:parameter nSteps: number of time steps -> durationTStep h/a

    	:return the fuel costs in €

    '''
    return disp*durationTStep(nSteps)*fuelC/(efficiency/100)

def calcOperCosts(disp, operC, nSteps):
    '''
    	:parameter disp: dispatched Power in MW
    	:parameter operC: fuel costs in €/MWhel
    	:parameter nSteps: number of time steps -> durationTStep h/a

    	:return the operating costs in €

    '''
    return disp*durationTStep(nSteps)*operC

def calcCapitalValue(KInv, KFuel, KOper, KCO2, RBF, nYears, lifetime, interestRate):
    '''
    	:parameter KInv: investment costs in €
    	:parameter KFuel: fuel costs in €/a
    	:parameter KOper: operating costs in €/a
    	:parameter KCO2: CO2-costs in €/a
    	:parameter RBF: discounting factor ("Rentenbarwertfaktor") in a

    	:return the captial value in €

    '''
    restValue = KInv * (lifetime - nYears) / lifetime

    return -float(KInv) - float(KFuel)*RBF - float(KOper)*RBF - float(KCO2)*RBF + float(restValue) / (1+interestRate/100)**nYears

def calcDiscCostsPerYear(KInv, KFuel, KOper, KCO2, RBF, nYears):
    # This function calculates the discounted costs per year (see calcCapitalValue-function for more information)
    return -KInv/RBF - (KFuel + KOper + KCO2)/nYears

def calcElecPrices(gas, ccGas, lignite, blackCoal, nuclear, wind, pv, nSteps, CO2Cost,
                   gasProps, ccGasProps, ligniteProps, blackCoalProps, nuclearProps, windProps, pvProps):
    '''
    	calculates the electricity prices over the whole time for each time step
    	:parameter gas, ccGas, ..., nuclear: dispatched power for each technology
    	:parameter nSteps: integer value number of time steps (10 or 8760 time steps per year)
    	:parameter CO2Cost: integer value for costs for CO2-emissions in €/t
    	:parameter gasProps, ccGasProps, ligniteProps, ..., pvProps: np-array for properties of different technologies
				   xxProps[0]: integer value for investment costs in €/kW
				   xxProps[1]: integer value for lifetime in years
				   xxProps[2]: float value for fuel costs in €/MWh_therm (=0 for Wind & PV)
				   xxProps[3]: float value for operating costs in €/MWh_el (=0 for Wind & PV)
				   xxProps[4]: integer value for efficiency in % (value for Wind & PV by renewables share .csv)
				   xxProps[5]: float value for specific CO2-emissions in kg/MWh_therm (=0 for Nuclear, Wind & PV)

    	:return list of electricity prices for each time step in €/MWh

    '''

    elecPrices = []
    dispPow = np.array((gas, ccGas, lignite, blackCoal, nuclear, wind, pv))

    for i in range(nSteps):

        # marginal costs are calculated for each technology:
        # marginalCosts[x] = kOper[x] + (kFuel[x] + kCO2 / 1000 * spEmissCO2[x]) / efficiency[x] * 100
        marginalCosts = [gasProps[3] + (gasProps[2] + CO2Cost / 1000 * gasProps[5]) / gasProps[4] * 100,
                         ccGasProps[3] + (ccGasProps[2] + CO2Cost / 1000 * ccGasProps[5]) / ccGasProps[4] * 100,
                         ligniteProps[3] + (ligniteProps[2] + CO2Cost / 1000 * ligniteProps[5]) / ligniteProps[4] * 100,
                         blackCoalProps[3] + (blackCoalProps[2] + CO2Cost / 1000 * blackCoalProps[5]) / blackCoalProps[4] * 100,
                         nuclearProps[3] + nuclearProps[2] / nuclearProps[4] * 100,
                         windProps[3],
                         pvProps[3]]

        # technolgies which do not supply energy in one time period are deleted for this period
        for k in range(7):
            if dispPow[k, i] == 0:
                marginalCosts[k] = 0
            else:
                pass

        # price for electricity equalized marginal costs of most expensive plant type used at that time
        elecPrices.append(max(marginalCosts))

    return elecPrices

# get functions

def getInstalledCap(myMD, Plant=None):
    '''
    	gets the installed Capacity for one of the power plant types or for all plants (if Plant=None)
    	:parameter myMD: function ModelProperties.py
    	:parameter Plant: string: ("Gas", "CCGas", "Lignite", "BlackCoal", "Nuclear", "Wind", "PV") or None

    	:return integer value of the installed capacity in MW

    '''
    if Plant==None:
        instCap = int(myMD.home.plantGroups["AllPlants"].installedCap.result[0])
    else:
        instCap = int(myMD.home.plantGroups["AllPlants"].plants[Plant].installedCap.result[0])

    return instCap

def getDispatchedPower(myMD, Plant=None):
    '''
    	gets the dispatched power over time for one of the power plant types or for all plants (if Plant=None)
    	:parameter myMD: function ModelProperties.py
    	:parameter Plant: string: ("Gas", "CCGas", "Lignite", "BlackCoal", "Nuclear", "Wind", "PV") or None

    	:return a list of values for the dispatched power over time in MW

    '''
    if Plant==None:
        dispPow = myMD.home.plantGroups["AllPlants"].dispatchedPower.result
    else:
        dispPow = myMD.home.plantGroups["AllPlants"].plants[Plant].dispatchedPower.result

    return dispPow

def getCO2Emissions(myMD, nSteps, specCO2emiss=None, efficiency=None, Plant=None):
    '''
    	gets the CO2-emission over the whole time for one of the power plant types or for all plants (if Plant=None)
    	:parameter myMD: function ModelProperties.py
    	:parameter specCO2emiss: in €/MWhtherm
    	:parameter Plant: string: ("Gas", "CCGas", "Lignite", "BlackCoal", "Nuclear", "Wind", "PV") or None

    	emissions = sum(power(t) * t * specCO2emiss / (efficiency/100))

    	:return integer of the CO2 emissions in Mt

    '''

    t = durationTStep(nSteps=nSteps)

    if Plant==None:
        emissions = int(myMD.home.emissionsCO2.result[0])
    elif Plant=="Gas":
        emissions = sum(multiList(myMD.home.plantGroups["AllPlants"].plants[Plant].dispatchedPower.result,
                                  t*specCO2emiss/(efficiency/100)))
    elif Plant=="CCGas":
        emissions = sum(multiList(myMD.home.plantGroups["AllPlants"].plants[Plant].dispatchedPower.result,
                                  t*specCO2emiss/(efficiency/100)))
    elif Plant=="BlackCoal":
        emissions = sum(multiList(myMD.home.plantGroups["AllPlants"].plants[Plant].dispatchedPower.result,
                                  t*specCO2emiss/(efficiency/100)))
    elif Plant=="Lignite":
        emissions = sum(multiList(myMD.home.plantGroups["AllPlants"].plants[Plant].dispatchedPower.result,
                                  t*specCO2emiss/(efficiency/100)))

    return emissions

# plot/charts functions

def piePlot(gas, ccGas, lignite, blackCoal, nuclear, wind, pv, caseID, language_strings, plotType='Cap'):
    '''
    	creates a pie plot for installed capacity or CO2-emissions
    	:parameter gas, ccGas, ..., nuclear: list with one value for each technology (installed capacity or CO2-emissions)
    	:parameter caseID: string of the case ID for the simulated output (path to output)
    	:parameter plotType Installed Capacity = 'Cap' or CO2-emissions = 'CO2'

    '''

  
    plt.rcParams['font.size'] = 36.0
    rc('font', **{'family': 'serif', 'serif': ['Arial']})

    figure(1, figsize=(26, 20))
    fig, ax = plt.subplots(figsize=(26, 20), subplot_kw=dict(aspect="equal"))

    data = [gas, ccGas, lignite, blackCoal, nuclear, wind, pv]
    sumAll = sum(data)
    colors = ['royalblue', 'cyan', 'peru', 'black', 'gold', 'green', 'red']

    if plotType=='Cap':
        labels = [string('lbl_chart_Gas',language_strings) + '\n' + str("%.1f" % (data[0]/1e3)) + 'GW' + '\n' + str("%.1f" % (data[0]/sumAll*100)) + '%', 
                  string('lbl_chart_GuD',language_strings) + '\n' + str("%.1f" % (data[1]/1e3)) + 'GW' + '\n' + str("%.1f" % (data[1]/sumAll*100)) + '%',
                  string('lbl_chart_Braunkohle',language_strings) + '\n' + str("%.1f" % (data[2]/1e3)) + 'GW' + '\n' + str("%.1f" % (data[2]/sumAll*100)) + '%', 
                  string('lbl_chart_Steinkohle',language_strings) + '\n' + str("%.1f" % (data[3]/1e3)) + 'GW' + '\n' + str("%.1f" % (data[3]/sumAll*100)) + '%',
                  string('lbl_chart_Nuklear',language_strings) + '\n' + str("%.1f" % (data[4]/1e3)) + 'GW' + '\n' + str("%.1f" % (data[4]/sumAll*100)) + '%', 
                  string('lbl_chart_Wind',language_strings) + '\n' + str("%.1f" % (data[5]/1e3)) + 'GW' + '\n' + str("%.1f" % (data[5]/sumAll*100)) + '%',
                  string('lbl_chart_PV',language_strings) + '\n' + str("%.1f" % (data[6]/1e3)) + 'GW' + '\n' + str("%.1f" % (data[6]/sumAll*100)) + '%']
    elif plotType=='CO2':
        labels = [string('lbl_chart_Gas',language_strings) + '\n' + str("%.1f" % (data[0]/sumAll*100)) + '%', 
                  string('lbl_chart_GuD',language_strings) + '\n' + str("%.1f" % (data[1]/sumAll*100)) + '%',
                  string('lbl_chart_Braunkohle',language_strings) + '\n' + str("%.1f" % (data[2]/sumAll*100)) + '%', 
                  string('lbl_chart_Steinkohle',language_strings) + '\n' + str("%.1f" % (data[3]/sumAll*100)) + '%',
                  string('lbl_chart_Nuklear',language_strings) + '\n' + str("%.1f" % (data[4]/sumAll*100)) + '%', 
                  string('lbl_chart_Wind',language_strings) + '\n' + str("%.1f" % (data[5]/sumAll*100)) + '%',
                  string('lbl_chart_PV',language_strings) + '\n' + str("%.1f" % (data[6]/sumAll*100)) + '%']


    i = 0

    while i < len(data):
        if data[i]/sumAll<=0.00001:
            del data[i]
            del colors[i]
            del labels[i]
        else:
            i += 1

    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40, colors=colors)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(labels[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    if plotType == 'Cap':
        ax.set_title(string('hd_piechart_InstallierteKapazität',language_strings), y=1.08)
        plt.savefig(caseID + 'charts/' + 'CapacityShare')
    elif plotType == 'CO2':
        ax.set_title(string('hd_piechart_CO2EmissionsAnteile',language_strings), y=1.08)
        plt.savefig(caseID + 'charts/' + 'CO2emissions')

    plt.close()

def barChart(gas, ccGas, lignite, blackCoal, nuclear, wind, pv, caseID, nYears, nSteps, language_strings):

    plt.rc('legend', **{'fontsize': 36})
    rc('font', **{'family': 'serif', 'serif': ['Arial']})

    figure(1, figsize=(26, 20))

    MW2TWh = durationTStep(nSteps) * 1e-6   #convert MW to TWh

    if nYears==1:

        if nSteps == 10:
            data = np.array((multiList(gas, MW2TWh), multiList(ccGas, MW2TWh), multiList(lignite, MW2TWh),
                             multiList(blackCoal, MW2TWh), multiList(nuclear, MW2TWh), multiList(wind, MW2TWh),
                             multiList(pv, MW2TWh)))
            index = np.arange(len(gas))
        else:
            data = np.array((reduceList(multiList(gas, MW2TWh),nSteps), reduceList(multiList(ccGas, MW2TWh),nSteps),
                             reduceList(multiList(lignite, MW2TWh),nSteps), reduceList(multiList(blackCoal, MW2TWh),nSteps),
                             reduceList(multiList(nuclear, MW2TWh),nSteps), reduceList(multiList(wind, MW2TWh),nSteps),
                             reduceList(multiList(pv, MW2TWh),nSteps)))
            index = np.arange(len(reduceList(divList(gas, MW2TWh), nSteps)))
        plotname = string('hd_barChart_GelieferteEnergieüberdieZeit',language_strings)

    else:

        if nSteps == 10:
            data = np.array((multiList(gas[0:nSteps], MW2TWh), multiList(ccGas[0:nSteps], MW2TWh),
                             multiList(lignite[0:nSteps], MW2TWh), multiList(blackCoal[0:nSteps], MW2TWh),
                             multiList(nuclear[0:nSteps], MW2TWh), multiList(wind[0:nSteps], MW2TWh),
                             multiList(pv[0:nSteps], MW2TWh)))
            index = np.arange(len(gas[0:nSteps]))
        else:
            data = np.array((reduceList(multiList(gas[0:nSteps], MW2TWh),nSteps),
                             reduceList(multiList(ccGas[0:nSteps], MW2TWh),nSteps),
                             reduceList(multiList(lignite[0:nSteps], MW2TWh),nSteps),
                             reduceList(multiList(blackCoal[0:nSteps], MW2TWh),nSteps),
                             reduceList(multiList(nuclear[0:nSteps], MW2TWh),nSteps),
                             reduceList(multiList(wind[0:nSteps], MW2TWh),nSteps),
                             reduceList(multiList(pv[0:nSteps], MW2TWh),nSteps)))
            index = np.arange(len(reduceList(multiList(gas[0:nSteps], MW2TWh), nSteps)))
        plotname = string('hd_barChart_GelieferteEnergieimerstenJahr',language_strings)

    colors = ['royalblue', 'cyan', 'peru', 'black', 'gold', 'green', 'red']
    labels = [string('lbl_chart_Gas',language_strings), 
              string('lbl_chart_GuD',language_strings), 
              string('lbl_chart_Braunkohle',language_strings), 
              string('lbl_chart_Steinkohle',language_strings), 
              string('lbl_chart_Nuklear',language_strings), 
              string('lbl_chart_Wind',language_strings), 
              string('lbl_chart_PV',language_strings)]


    bar_width = 0.1

    i = 0

    while i < len(data):

        if sum(data[i]) <= 0.00001:
            data = np.delete(data, i, 0)
            del colors[i]
            del labels[i]

        else:

            plt.bar(index + bar_width*i,
                    data[i],
                    bar_width,
                    color=colors[i],
                    label=labels[i])
            i +=1

    plt.subplot(111).spines["top"].set_visible(False)
    plt.subplot(111).spines["bottom"].set_visible(False)
    plt.subplot(111).spines["right"].set_visible(False)
    plt.subplot(111).spines["left"].set_visible(False)

    for y in range(5, int(np.amax(data))+5, 5):
        plt.plot(range(0, 11), [y] * len(range(0, 11)), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    plt.title(plotname)
    plt.xlabel(string('lbl_barChart_xAchse',language_strings))
    plt.ylabel(string('lbl_barChart_yAchse',language_strings))
    
    plt.xticks(index + bar_width / 2 * len(data), ('t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10'))
    plt.yticks(range(0, int(np.amax(data))+5, 5))
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=6)
    art.append(lgd)

    plt.tight_layout()
    plt.savefig(caseID + 'charts/' + 'DispatchedPower', additional_artists=art, bbox_inches="tight")

    plt.close()

def supplyProfileChart(gas, ccGas, lignite, blackCoal, nuclear, wind, pv, caseID, nYears, nSteps, language_strings):

    plt.rc('legend', **{'fontsize': 36})
    rc('font', **{'family': 'serif', 'serif': ['Arial']})
    figure(1, figsize=(26, 20))

    data = np.array((divList(gas, 1e3), divList(ccGas, 1e3), divList(lignite, 1e3), divList(blackCoal, 1e3),
                     divList(nuclear, 1e3), divList(wind, 1e3), divList(pv, 1e3)))
    colors = ['royalblue', 'cyan', 'peru', 'black', 'gold', 'green', 'red']
    labels = [string('lbl_chart_Gas',language_strings), 
              string('lbl_chart_GuD',language_strings),
              string('lbl_chart_Braunkohle',language_strings), 
              string('lbl_chart_Steinkohle',language_strings), 
              string('lbl_chart_Nuklear',language_strings), 
              string('lbl_chart_Wind',language_strings), 
              string('lbl_chart_PV',language_strings)]

    i=0

    while i < len(data):

        if sum(data[i]) <= 0.00001:
            data = np.delete(data, i, 0)
            del colors[i]
            del labels[i]

        else:
            if nSteps != 8760:
                plt.plot(np.arange(0,nYears*8760,8760/nSteps), data[i], linewidth=5, color=colors[i], label=labels[i])
            else:
                plt.plot(data[i], linewidth=5, color=colors[i], label=labels[i])
            i+=1

    plt.subplot(111).spines["top"].set_visible(False)
    plt.subplot(111).spines["bottom"].set_visible(False)
    plt.subplot(111).spines["right"].set_visible(False)
    plt.subplot(111).spines["left"].set_visible(False)

    #for y in range(5, int(np.amax(data))+5, 5):
     #   plt.plot(range(0, nYears*8760), [y] * len(range(0, nYears*8760)), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    #legend = ax.legend(loc='upper right', shadow=False, fontsize='x-large')
    
    plt.title(string('hd_supplyProfileChart_Leistungsprofil',language_strings))
    plt.xlabel(string('lbl_supplyProfileChart_xAchse',language_strings))
    plt.ylabel(string('lbl_supplyProfileChart_yAchse',language_strings))
    
    plt.xticks()
    plt.yticks(range(0, int(np.amax(data))+5, 5))
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=6)
    art.append(lgd)

    plt.tight_layout()
    plt.savefig(caseID + 'charts/' + 'SupplyProfile', additional_artists=art, bbox_inches="tight")

    plt.close()

def demandChart(demand, nSteps, nYears, caseID, language_strings):

    plt.rc('legend', **{'fontsize': 36})
    rc('font', **{'family': 'serif', 'serif': ['Arial']})
    figure(1, figsize=(26, 20))

    if nSteps == 10:
        plt.plot(np.arange(0,nYears*8760,8760/nSteps), divList(demand, 1e3), linewidth=5, color='black', label='')
        title = string('hd_demandChart_Jahresdauerlinie',language_strings)
    else:
        plt.plot(divList(demand, 1e3), linewidth=5, color='black', label='')
        title = string('hd_demandChart_Nachfrageprofil',language_strings)

    plt.subplot(111).spines["top"].set_visible(False)
    plt.subplot(111).spines["bottom"].set_visible(False)
    plt.subplot(111).spines["right"].set_visible(False)
    plt.subplot(111).spines["left"].set_visible(False)

    for y in range(10, 80, 10):
        plt.plot(range(0, 8760), [y] * len(range(0, 8760)), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    plt.title(title)
    plt.xlabel(string('lbl_demandChart_xAchse',language_strings))
    plt.ylabel(string('lbl_demandChart_yAchse',language_strings))

    plt.xticks()
    plt.yticks()
    plt.ylim(ymin=0)

    plt.tight_layout()
    plt.savefig(caseID + 'charts/' + 'DemandProfile')

    plt.close()

def priceChart(elecPrices, nSteps, nYears, caseID, language_strings):

    plt.rc('legend', **{'fontsize': 36})
    rc('font', **{'family': 'serif', 'serif': ['Arial']})
    figure(1, figsize=(26, 20))

    if nSteps == 10:
        plt.plot(np.arange(0,nYears*8760,8760/nSteps), elecPrices, linewidth=5, color='black', label='')
    else:
        plt.plot(elecPrices, linewidth=5, color='black', label='')

    plt.subplot(111).spines["top"].set_visible(False)
    plt.subplot(111).spines["bottom"].set_visible(False)
    plt.subplot(111).spines["right"].set_visible(False)
    plt.subplot(111).spines["left"].set_visible(False)

    for y in range(10, 80, 10):
        plt.plot(range(0, 8760), [y] * len(range(0, 8760)), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    plt.title(string('hd_priceChart_StromPreisMeritOrder',language_strings))
    plt.xlabel(string('lbl_priceChart_xAchse',language_strings))
    plt.ylabel(string('lbl_priceChart_yAchse',language_strings))
    
    plt.xticks()
    plt.yticks()
    plt.ylim(ymin=0)

    plt.tight_layout()
    plt.savefig(caseID + 'charts/' + 'ElectricityPrices')

    plt.close()