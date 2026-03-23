
import pandas as pd

def run_integrity_checks():
    # 1. Cargar solo los datos de entrenamiento
    print("--- INICIANDO CHEQUEOS DE INTEGRIDAD ---")
    try:
        df = pd.read_csv('data/processed/train.csv')
    except FileNotFoundError:
        print("Error: No se encontró data/interim/train.csv")
        return

    # 2. Chequeo de campos vacíos (Missing Values)
    nulos = df.isnull().sum().sum()
    print(f"[Check Nulos]: Cantidad total de campos vacíos en el dataset: {nulos}")

    # 3. Chequeos de rangos válidos (Meses y Clima)
    meses_validos = df['mnth'].between(1, 12).all()
    clima_valido = df['weathersit'].between(1, 4).all()
    estacion_valida = df['season'].between(1, 4).all()
    
    print(f"[Check Rangos]: ¿Todos los meses van de 1 a 12? {'Sí' if meses_validos else 'NO'}")
    print(f"[Check Rangos]: ¿Todas las estaciones van de 1 a 4? {'Sí' if estacion_valida else 'NO'}")
    print(f"[Check Rangos]: ¿Todo el clima va de 1 a 4? {'Sí' if clima_valido else 'NO'}")

    # 4. Verificar qué clases existen realmente en weathersit
    clases_clima = df['weathersit'].unique()
    clases_clima.sort()
    print(f"[Check Clases]: Tipos de clima presentes en los datos: {clases_clima}")

    # 5. Chequeo de integridad cruzada: workingday vs holiday/weekend
    # weekend suele ser weekday 0 (Domingo) y 6 (Sábado) según estándares, 
    # comprobaremos si hay algún workingday=1 que a la vez sea holiday=1
    errores_logicos = df[(df['workingday'] == 1) & (df['holiday'] == 1)]
    print(f"[Check Lógica]: Días marcados como 'Laborables' pero que también son 'Feriados': {len(errores_logicos)}")

    # 6. Chequeos estrictos de categorías
    yr_valido = df['yr'].between(0, 1).all()
    weekday_valido = df['weekday'].between(0, 6).all()
    
    print(f"\n[Check Categorías]: ¿Todo 'yr' es 0 o 1? {'Sí' if yr_valido else 'NO'}")
    print(f"[Check Categorías]: ¿Todo 'weekday' va de 0 a 6? {'Sí' if weekday_valido else 'NO'}")

    # 7. Chequeo de variables normalizadas (0 a 1)
    print("\n[Check Normalización]: Verificando variables continuas...")
    vars_normalizadas = ['temp', 'atemp', 'hum', 'windspeed']
    for var in vars_normalizadas:
        rango_valido = df[var].between(0, 1).all()
        print(f" - ¿'{var}' está estrictamente entre 0 y 1? {'Sí' if rango_valido else 'NO'}")
        
        # Si llega a dar falso, imprimimos los extremos para ver la gravedad del error
        if not rango_valido:
            print(f"   -> ALERTA en '{var}': El mínimo es {df[var].min()} y el máximo es {df[var].max()}")

    print("--- CHEQUEOS FINALIZADOS ---")

if __name__ == "__main__":
    run_integrity_checks()
