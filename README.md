# 72.75-machine-learning

Decidimos usar day.csv en lugar de hour.csv dado que nuestro objetivo sera determinar las rentas de bicicletas por dia. Ademas de que hour.csv es un dataset muy extenso y los metodos vistos y que vamos a aplicar tienen mayor sentido en datasets medianos, como day.csv.

Se uso integrity_checks.py para verificar la integridad de los datos.

# Outliers
Los outliers se visualizan corriendo boxplot_outliers.py

## Windspeed
En la documentacion no se especifica como se mide el viento, bajo que unidades, por ser Estados Unidos podemos asumir que la medicion es en millas por hora.

De los outliers el valor maximo que dio es 0.441563. Segun la documentacion la columna de windspeed se construye:
Normalized wind speed. The values are divided to 67 (max)

Es decir el valor maximo que nos dan los outliers es 29.584721 mph segun la [escala de beaufort](https://www.rmets.org/metmatters/beaufort-wind-scale) eso equivale a una brisa fuerte, por lo que no hay razon para considerar ilógicos estos valores.

Variable: windspeed  -> Outliers detectados: 11
windspeed: {
  434: 0.4148
  293: 0.422275
  451: 0.386821
  421: 0.421642
  94: 0.385571
  383: 0.415429
  433: 0.441563
  722: 0.407346
  95: 0.388067
  408: 0.409212
  667: 0.398008
}

## Humidity

Tenemos un solo valor de outlier de 0.0 en el instant 69, lo cual es raro. Se interpolo el valor usando impute_data.py. Se buscan vecinos con valores de temperatura y velocidad de viento similares para inferir la humedad.

## Test

Hay outliers para velocidad del viento y humedada en algunos de los campos pero no los podemos identificar como errores. Verficacamos para esos dias usando el archivo day.csv y los promedios dan lo que se muestra como outliers.

Variable: hum        -> Detected outliers: 1
hum: {
  50: 0.187917
}
Variable: windspeed  -> Detected outliers: 2
windspeed: {
  50: 0.507463
  45: 0.417908
}

# Variables Categóricas

# Limpieza de variables

## Variables que no ofrecen información.
### instant
Es un id de la row.

### yr
Nos habla del 2011 y 2012, no tiene sentido extrapolar esa información al resto de años sin tener en cuenta variables económicas de esos años.

### dteday
No nos da información valiosa, al usar el csv de day, cada info es diaria.

## Variables correlacionadas:
### temp con atemp
La temperatura y sensacion termica son valores que aportan practicamente la misma información lo que produce que esten altamente correlacionadas. Nos quedamos con atemp que es lo que sienten las personas realmente.

### Meses con Season
En boxplots_categorics_cnt.png vemos como se relacionan ambas variables en cuanto a los valores que producen para cnt.

Ademas ambas nos dan la misma informacion, en que epoca del año nos encontramos. La informacion de months nos permite inferir mas informacion mes a mes pero el problema es que al usar one-hot encoding tendremos 12 variables nuevas contra 4.

### Casual y Registered con cnt
Por mas que en la matriz de correlacion solo vemos cnt y registered como correlacionadas, no tiene sentido usarlas casual y registered para evaluar ya que por definicion se suman para obtener cnt que es nuestro target.
