# Gestion de posiciones abiertas por profit
# Gestion de posiciones abiertas por simbolo
# Gestion de posiciones abiertas por riesgo
# Gestion de posiciones abiertas por volatilidad

class MarketMasterManagement():
    def __init__(self, marketmaster):
        self.marketmaster = marketmaster
        super().__init__()

    def profit_management(self, args, sentido, ask, bid, data_5m_actual, spread):
        # esta  es para hacer sl/tp mas grande o pequeño en funcion de como se mueve el precio
        sl_multiplier = self.marketmaster.atr_multipliers["all_data_5m"][0]
        tp_multiplier = self.marketmaster.atr_multipliers["all_data_5m"][1]

        atr = data_5m_actual['atr']
        atr_ma = data_5m_actual['atr_ma']

        tp, sl, _ = self.marketmaster.liquidity_tp_sl(sentido, ask, bid, atr, atr_ma, spread, args["puntos_liquidez"].keys(), sl_multiplier, tp_multiplier)
        return tp, sl

    def symbol_management(self):
        pass

    def risk_management(self):
        pass

    def volatility_management(self):
        pass