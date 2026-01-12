import numpy as np

class MarketMasterStats:
    def __init__(self, corr_limit):
        self.corr_limit = corr_limit
        super().__init__()

    def fibonacci(self,data: np.array, fibonacci_values: tuple, price: float, sense: int) -> bool:
        """
        Calcula si el precio está dentro de una zona óptima de Fibonacci.

        Args:
            data (np.array): Serie de precios.
            fibonacci_values (tuple): Niveles de Fibonacci a considerar (ej. (0.236, 0.382, 0.5, 0.618, 0.886)).
            price (float): Precio a evaluar.
            sense (int): Dirección de la tendencia (1 para alcista, -1 para bajista).

        Returns:
            bool: True si el precio está en la zona óptima, False en caso contrario.
        """
        min_value = np.min(data)
        max_value = np.max(data)
        rango = max_value - min_value

        niveles = [max_value - rango * fib for fib in fibonacci_values] if sense == 1 else \
                [min_value + rango * fib for fib in fibonacci_values]

        return any(niveles[i] <= price <= niveles[i+1] for i in range(len(niveles) - 1))

    def correlation_confirmation(self,args,signal1,signal2,name2,lotaje,peso):
        if args["name"]!=name2 and args['correlation_dict'][f"{args["name"]}-{name2}"][0] > self.corr_limit:
            """if action==action2:
                lotaje=lotaje*2

            elif action==-action2:
                lotaje=lotaje/2"""
            
            if signal1 == -signal2:
                lotaje = lotaje * peso

        elif args["name"]!=name2 and args['correlation_dict'][f"{args["name"]}-{name2}"][0] < -self.corr_limit:
            """if action==-action2:
                lotaje=lotaje*2

            elif action==action2:
                lotaje=lotaje/2"""
            
            if signal1 == signal2:
                lotaje = lotaje * peso

        return lotaje