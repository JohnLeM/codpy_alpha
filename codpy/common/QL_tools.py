import QuantLib as ql
import numpy as np
import os,sys
from datetime import date,datetime,timedelta
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:sys.path.append(parentdir)



def VanillaOption(today_date, 
                    spot_price, 
                    risk_free_rate, 
                    dividend_rate, 
                    strike_price, 
                    volatility, 
                    maturity_date,
                    day_count = ql.Actual365Fixed(),
                    calendar = ql.UnitedKingdom(),
                    option_type = ql.Option.Call):

    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, exercise)    

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    # calculation_date = date.fromtimestamp(today_date)
    # ql_calculation_date = ql.Date(calculation_date.day,calculation_date.month,calculation_date.year)
    ql.Settings.instance().evaluationDate = today_date
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today_date, risk_free_rate, day_count))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(today_date, dividend_rate, day_count))
    flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today_date, calendar, volatility, day_count))
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, 
                                            dividend_yield, 
                                            flat_ts, 
                                            flat_vol_ts)  
    european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
    out = european_option.NPV()
    return out

def PNLVanillaOption(PricesDates, Spotprices, ShiftPricesDates, ShiftSpotprices, risk_free_rate  = 0.0, dividend_rate  = 0.01, strike_price  = 3000, volatility = .1, maturity_date = ql.Date(1, 1, 2023), notional = 10000.):
    out = np.zeros(len(PricesDates))

    for i in range(len(PricesDates)) :
        datei = PricesDates[i]
        shiftdatei = ShiftPricesDates[i]
        spoti = Spotprices[i]
        shiftspoti = ShiftSpotprices[i]
        shiftvaluei = VanillaOption(shiftdatei, shiftspoti, risk_free_rate, dividend_rate, strike_price, volatility, maturity_date)
        valuei = VanillaOption(datei, spoti, risk_free_rate, dividend_rate, strike_price, volatility, maturity_date)
        out[i] = shiftvaluei - valuei
    return out*notional

