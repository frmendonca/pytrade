from typing import List

class DiscountCashFlowModel:
    def __init__(self):
        ...

    def compute_dcf_model(
        #self,
        fcf_forecast: List[float],
        discount_rate: float,
        terminal_growth_rate: float
    ) -> float:

        T = len(fcf_forecast)

        # Discount cash flows
        dfcf = [fcf / (1 + discount_rate) ** (t+1) for t, fcf in enumerate(fcf_forecast)]

        # Terminal value
        tv = fcf_forecast[-1] * (1 + terminal_growth_rate)/(discount_rate - terminal_growth_rate)
        dtv = tv / (1 + discount_rate) ** T

        # DCF_value
        dcf_value = sum(dfcf) + dtv

        return dcf_value


