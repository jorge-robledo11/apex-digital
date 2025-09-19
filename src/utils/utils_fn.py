# • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ Pyfunctions ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ •
# Librerías y/o depedencias
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import empiricaldist
sns.set_theme(context='notebook', style=plt.style.use('dark_background'))

# Capturar variables
# Función para capturar los tipos de variables
def capture_variables(data: pd.DataFrame) -> tuple:
    
    """
    Function to capture the types of Dataframe variables

    Args:
        dataframe: DataFrame
    
    Return:
        variables: A tuple of lists
    
    The order to unpack variables:
    1. numericals
    2. continous
    3. categoricals
    4. discretes
    5. temporaries
    """

    numericals = list(data.select_dtypes(include=[np.int32, np.int64, np.float32, np.float64]).columns)
    discretes = [col for col in data[numericals] if len(data[numericals][col].unique()) <= 5]
    temporaries = list(data.select_dtypes(include=['datetime', 'timedelta']).columns)
    categoricals = list(data.select_dtypes(include=['category', 'object', 'bool']).columns)
    continuous = [col for col in data[numericals] if col not in discretes]
    
    variables = tuple((continuous, categoricals, discretes, temporaries))

    # Retornamos una tupla de listas
    return variables
            

# Función para graficar los datos con valores nulos
def plotting_nan_values(data: pd.DataFrame) -> None:

    """
    Function to plot nan values

    Args:
        data: DataFrame
    
    Return:
        Dataviz
    """

    vars_with_nan = [var for var in data.columns if data[var].isnull().sum() > 0]
    
    if len(vars_with_nan) == 0:
        print('No se encontraron variables con nulos')
    
    else:
        # Plotting
        plt.figure(figsize=(14, 6))
        data[vars_with_nan].isnull().mean().sort_values(ascending=False).plot.bar(color='crimson', width=0.4, 
                                                                                  edgecolor='skyblue', lw=0.75)
        plt.axhline(1/3, color='#E51A4C', ls='dashed', lw=1.5, label='⅓ Missing Values')
        plt.ylim(0, 1)
        plt.xlabel('Predictors', fontsize=12)
        plt.ylabel('Percentage of missing data', fontsize=12)
        plt.xticks(fontsize=10, rotation=25)
        plt.yticks(fontsize=10)
        plt.legend()
        plt.grid(color='white', linestyle='-', linewidth=0.1)
        plt.tight_layout()


# Variables estratificadas por clases
# Función para obtener la estratificación de clases/target
def class_distribution(data: pd.DataFrame, target: str) -> None:
    
    """
    Function to get balance by classes

    Args:
        data: DataFrame
        target: str
    
    Return:
        Dataviz
    """

    # Distribución de clases
    distribucion = data[target].value_counts(normalize=True)

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 4))

    # Ajustar el margen izquierdo de los ejes para separar las barras del eje Y
    ax.margins(y=0.2)

    # Ajustar la posición de las etiquetas de las barras
    ax.invert_yaxis()

    # Crear gráfico de barras horizontales con la paleta de colores personalizada
    ax.barh(
        y=distribucion.index, 
        width=distribucion.values, 
        align='center', 
        color='#9fa8da',
        edgecolor='white', 
        height=0.5, 
        linewidth=0.5
    )

    # Definir título y etiquetas de los ejes
    ax.set_title('Distribución de clases\n', fontsize=12)
    ax.set_xlabel('Porcentajes', fontsize=10)
    ax.set_ylabel(f'{target}'.capitalize(), fontsize=10)

    # Mostrar el gráfico
    plt.grid(color='white', linestyle='-', linewidth=0.1)
    plt.tight_layout()
    plt.show()


# Función para obtener la matriz de correlaciones entre los predictores
def continuous_correlation_matrix(data: pd.DataFrame, continuous: list) -> None:
    
    """
    Function to plot correlation_matrix

    Args:
        data: DataFrame
        continuous: list
    
    Return:
        Dataviz
    """
    
    correlations = data[continuous].corr(method='pearson', numeric_only=True)
    plt.figure(figsize=(17, 10))
    sns.heatmap(correlations, vmax=1, annot=True, cmap='gist_yarg', linewidths=1, square=True)
    plt.title('Matriz de Correlaciones\n', fontsize=12)
    plt.xticks(fontsize=10, rotation=25)
    plt.yticks(fontsize=10, rotation=25)
    plt.tight_layout()


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calcula la asociación entre dos variables categóricas usando Cramér's V.
    
    Parámetros:
    -----------
    x, y: arrays (o Series de pandas) con datos categóricos de la misma longitud.

    Retorna:
    --------
    float: valor de Cramér's V (entre 0 y 1).
    """
    # Tabla de contingencia
    contingency_table = pd.crosstab(x, y)

    # Chi-Cuadrado
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    # Número de observaciones
    n = contingency_table.sum().sum()

    # Valor de Cramér's V
    phi2 = chi2/n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    if denom == 0:
        return 0.0

    return np.sqrt(phi2corr / denom)


def categoricals_correlation_matrix(data: pd.DataFrame, categoricals: list) -> None:
    """
    Crea un heatmap de la asociación (Cramér's V) entre variables categóricas.

    Parámetros:
    -----------
    data : pd.DataFrame
        DataFrame que contiene las variables categóricas.
    categoricals : list
        Lista con los nombres de las columnas categóricas.

    Retorna:
    --------
    None (muestra un gráfico heatmap con la matriz de asociaciones).
    """
    # Inicializar la matriz
    n = len(categoricals)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), index=categoricals, columns=categoricals)

    # Calcular Cramér's V para cada par de variables
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
            elif i < j:
                val = cramers_v(data[categoricals[i]], data[categoricals[j]])
                corr_matrix.iloc[i, j] = val
                corr_matrix.iloc[j, i] = val

    # Graficar la matriz con Seaborn
    plt.figure(figsize=(17, 10))
    sns.heatmap(corr_matrix, annot=True, vmin=0, vmax=1, cmap='YlGnBu', linewidths=1, square=True)
    plt.title("Matriz de Asociación Cramér's V", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Diagnóstico de variables
# Función para observar el comportamiento de variables continuas
def diagnostic_plots(data: pd.DataFrame, variables: list) -> None:

    """
    Function to get diagnostic graphics into 
    numerical (continous and discretes) predictors

    Args:
        data: DataFrame
        variables: list
    
    Return:
        Dataviz
    """
        
    for var in data[variables]:
        fig, (ax, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle('Diagnostic Plots', fontsize=16)
        plt.rcParams.update({'figure.max_open_warning': 0}) # Evitar un warning

        # Histogram Plot
        plt.subplot(1, 4, 1)
        plt.title('Histogram Plot')
        sns.histplot(data[var], bins=25, color='midnightblue', edgecolor='white', lw=0.5)
        plt.axvline(data[var].mean(), color='#E51A4C', ls='dashed', lw=1.5, label='Mean')
        plt.axvline(data[var].median(), color='gold', ls='dashed', lw=1.5, label='Median')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=25)
        plt.xlabel(var)
        plt.grid(color='white', linestyle='-', linewidth=0.1)
        plt.legend(fontsize=10)
        
        # CDF Plot
        plt.subplot(1, 4, 2)
        plt.title('CDF Plot')
        xs = np.linspace(data[var].min(), data[var].max())
        ys = stats.norm(data[var].mean(), data[var].std()).cdf(xs) # Distribución normal a partir de unos datos
        plt.plot(xs, ys, color='cornflowerblue', ls='dashed')
        empiricaldist.Cdf.from_seq(data[var], normalize=True).plot(color='chartreuse')
        plt.xlabel(var)
        plt.xticks(rotation=25)
        plt.ylabel('Probabilidad')
        plt.legend(['Distribución normal', var], fontsize=8, loc='upper left')
        plt.grid(color='white', linestyle='-', linewidth=0.1)

        # PDF Plot
        plt.subplot(1, 4, 3)
        plt.title('PDF Plot')
        kurtosis = stats.kurtosis(data[var], nan_policy='omit') # Kurtosis
        skew = stats.skew(data[var], nan_policy='omit') # Sesgo
        xs = np.linspace(data[var].min(), data[var].max())
        ys = stats.norm(data[var].mean(), data[var].std()).pdf(xs) # Distribución normal a partir de unos datos
        plt.plot(xs, ys, color='cornflowerblue', ls='dashed')
        sns.kdeplot(data=data, x=data[var], fill=True, lw=0.75, color='crimson', alpha=0.5, edgecolor='white')
        plt.text(s=f'Skew: {skew:0.2f}\nKurtosis: {kurtosis:0.2f}',
                 x=0.25, y=0.65, transform=ax3.transAxes, fontsize=11,
                 verticalalignment='center', horizontalalignment='center')
        plt.ylabel('Densidad')
        plt.xticks(rotation=25)
        plt.xlabel(var)
        plt.xlim()
        plt.legend(['Distribución normal', var], fontsize=8, loc='upper right')
        plt.grid(color='white', linestyle='-', linewidth=0.1)

        # Boxplot & Stripplot
        plt.subplot(1, 4, 4)
        plt.title('Boxplot')
        sns.boxplot(data=data[var], width=0.4, color='silver',
                    boxprops=dict(lw=1, edgecolor='white'),
                    whiskerprops=dict(color='white', lw=1),
                    capprops=dict(color='white', lw=1),
                    medianprops=dict(),
                    flierprops=dict(color='red', lw=1, marker='o', markerfacecolor='red'))
        plt.axhline(data[var].quantile(0.75), color='magenta', ls='dotted', lw=1.5, label='IQR 75%')
        plt.axhline(data[var].median(), color='gold', ls='dashed', lw=1.5, label='Median')
        plt.axhline(data[var].quantile(0.25), color='cyan', ls='dotted', lw=1.5, label='IQR 25%')
        plt.xlabel(var)
        plt.tick_params(labelbottom=False)
        plt.ylabel('Unidades')
        plt.legend(fontsize=8, loc='upper right')
        plt.grid(color='white', linestyle='-', linewidth=0.1)
        fig.tight_layout()
        

# Función para graficar las variables categóricas
def categoricals_plot(data: pd.DataFrame, variables: list) -> None:
    
    """
    Function to get distributions graphics into 
    categoricals and discretes predictors

    Args:
        data: DataFrame
        variables: list
    
    Return:
        Dataviz
    """
    
    # Definir el número de filas y columnas para organizar los subplots
    num_rows = (len(variables) + 1) // 2  # Dividir el número de variables por 2 y redondear hacia arriba
    num_cols = 2  # Dos columnas de gráficos por fila

    # Crear una figura y ejes para organizar los subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(28, 36))
    
    plt.suptitle('Categoricals Plots\n', fontsize=24, y=0.95)
    
    # Asegurarse de que 'axes' sea una matriz 2D incluso si solo hay una variable
    if len(variables) == 1:
        axes = axes.reshape(1, -1)

    # Iterar sobre las variables y crear gráficos para cada una
    for i, var in enumerate(variables):
        row, col = i // 2, i % 2  # Calcular la fila y columna actual

        # Crear un gráfico de barras en los ejes correspondientes
        temp_dataframe = pd.Series(data[var].value_counts(normalize=True))
        temp_dataframe.sort_values(ascending=False).plot.bar(color='#f19900', edgecolor='skyblue', ax=axes[row, col])
        
        # Añadir una línea horizontal a 5% para resaltar las categorías poco comunes
        axes[row, col].axhline(y=0.05, color='#E51A4C', ls='dashed', lw=1.5)
        axes[row, col].set_ylabel('Porcentajes')
        axes[row, col].set_xlabel(var)
        axes[row, col].set_xticklabels(temp_dataframe.index, rotation=25)
        axes[row, col].grid(color='white', linestyle='-', linewidth=0.25)
    
    # Ajustar automáticamente el espaciado entre subplots
    plt.grid(color='white', linestyle='-', linewidth=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=0.2)  # El argumento rect controla el espacio para el título superior
    plt.show()


# Función para graficar las categóricas segmentadas por el target
def categoricals_hue_target(data: pd.DataFrame, variables: list, target: str) -> None:
    
    # Graficos de cómo covarian algunas variables con respecto al target
    paletas = ['rocket', 'mako', 'crest', 'magma', 'viridis', 'flare']
    np.random.seed(11)

    for var in data[variables]:
        plt.figure(figsize=(12, 6))
        plt.title(f'{var} segmentado por {target}\n', fontsize=12)
        sns.countplot(x=var, hue=target, data=data, edgecolor='white', lw=0.5, palette=np.random.choice(paletas))
        plt.ylabel('Cantidades')
        plt.xticks(fontsize=12, rotation=25)
        plt.yticks(fontsize=12)
        plt.grid(color='white', linestyle='-', linewidth=0.1)
        plt.tight_layout()
