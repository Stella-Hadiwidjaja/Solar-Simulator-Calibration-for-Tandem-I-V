import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import minimize
import json
import warnings
import pickle

def import_data(file_path,sep=','):
    """
    Imports EQE sub-cell data

    Parameters
    ----------
    file_path: str
        file path to sub-cell data
    sep = str, default: ','
        data seperator (comma, space, tab, etc)
    """
    df = pd.read_csv(file_path, sep=sep)
    #df = df.rename(columns={'Wavelength (nm)':'Wavelength (nm)', 'EQE (%)':'EQE (%)', 'SR (A/W)':'SR (A/W)'})

    # Convert all columns to float64
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    return df


def plot_EQE(PVSK, Si, color1='blue', color2='red', wavelength_lim = [300,1200],split = False):
    """
    Plots EQE of two-sub cells, either combined in a single plot or displayed as separate subplots.
    
    Parameters
    ----------
    PVSK : pandas.DataFrame
        DataFrame containing EQE data for the perovskite top cell.
        Must contain columns:
        - 'Wavelength (nm)' : Wavelength values in nanometers
        - 'EQE (%)' : External Quantum Efficiency values in percentage
    
    Si : pandas.DataFrame
        DataFrame containing EQE data for the silicon bottom cell.
        Must contain columns:
        - 'Wavelength (nm)' : Wavelength values in nanometers
        - 'EQE (%)' : External Quantum Efficiency values in percentage
    
    color1 : str, optional, default: 'blue'
        Color for the PVSK top cell plot line and markers
    
    color2 : str, optional, default: 'red'
        Color for the Si bottom cell plot line and markers
    
    wavelength_lim : list, optional, default: [300, 1200]
        Wavelength range limits for the x-axis [min, max].
        Note: Currently defined but not implemented in the plotting.
    
    split : bool, optional, default: False
        If False, plots both EQE curves in a single combined plot.
        If True, creates two separate subplots side by side.
    
    Returns
    -------
    None
        Displays the matplotlib plot(s) but does not return any values.
    
    Examples
    --------
    >>> # Plot both cells in one figure
    >>> plot_EQE(pvsk_data, si_data)
    
    >>> # Plot cells in separate subplots with custom colors
    >>> plot_EQE(pvsk_data, si_data, color1='green', color2='orange', split=True)

    """
    if split == False:
        # Plot both junctions in the same plot with different colors
        plt.figure(figsize=(6, 3),dpi=300)
        plt.plot(PVSK['Wavelength (nm)'], PVSK['EQE (%)'], label='Top', color='blue', marker='o')
        plt.plot(Si['Wavelength (nm)'], Si['EQE (%)'], label='Bot', color='red', marker='x')
        #plt.title('EQE of Two Junctions (Same Plot)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('EQE (%)')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        # Plot both junctions in two different plots side by side
        fig, axes = plt.subplots(1, 2, figsize=(7, 4),dpi=300)

        # Plot for Sub Cell 1
        axes[0].plot(PVSK['Wavelength (nm)'], PVSK['EQE (%)'], color='blue', marker='o')
        #axes[0].set_title('PVSK EQE')
        axes[0].set_xlabel('Wavelength (nm)')
        axes[0].set_ylabel('EQE (%)')
        axes[0].set_ylim(0, 100)
        axes[0].grid(True)

        # Plot for Sub Cell 2
        axes[1].plot(Si['Wavelength (nm)'], Si['EQE (%)'], color='red', marker='x')
        #axes[1].set_title('Si EQE')
        axes[1].set_xlabel('Wavelength (nm)')
        axes[1].set_ylabel('EQE (%)')
        axes[1].set_ylim(0, 100)
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()


#This function create the Base spectrum of your reference spectrum
def fit_spectrum(E_ref, Solar_Simulator_LEDs_path, recipe_name,  beginning_w=350, end_w=1120, show_plot=True,wspc=True,initial_guess=None):
    """
    Approximates a target spectrum (E_ref) using a combination of LEDs and plots the result. 
    Parameters:
            E_ref (str): Path to the target/reference spectrum CSV file.
            Solar Simulator (int): Solar Simulator number to select LED configuration.
            recipe_name (str): Name of the output recipe (used for saving the .wspc file).
            beginning_w (int): Starting wavelength [nm]. Default is 350.
            end_w (int): Ending wavelength [nm]. Default is 1120.
            show_plot (bool): Whether to show the comparison plot.
            initial_guess: default: None - initial guess of {alpha} 

    Returns:
        pd.DataFrame: The base spectrum as a DataFrame with 'Wavelength (nm)' and 'Irradiance (W/m^2/nm)'.
    """

        # Load reference spectrum if a path was provided
    if isinstance(E_ref, str):
        if not os.path.exists(E_ref):
            raise FileNotFoundError(f"The reference spectrum file '{E_ref}' was not found.")
        E_ref = pd.read_csv(E_ref, delimiter=',')
    elif not isinstance(E_ref, pd.DataFrame):
        raise TypeError("E_ref must be either a file path (str) or a pandas DataFrame.")

    # Validate required columns
    required_cols = {"Wavelength (nm)", "Irradiance (W/m^2/nm)"}
    if not required_cols.issubset(E_ref.columns):
        raise ValueError(f"Reference spectrum must contain columns: {required_cols}")

    # Ensure wavelength is float
    E_ref["Wavelength (nm)"] = E_ref["Wavelength (nm)"].astype(float)

    # Get the LED configuration
    LEDs = get_LEDs(beginning_w, end_w, Solar_Simulator_LEDs_path)

    #Get the recipe and import the recipe into a spectrum file that can be used by wavelabs software. You can choose the name of the spectrum file
    non_linear_alphas = get_nonlinear_recipe(beginning_w, end_w, E_ref, LEDs,initial_guess=initial_guess)

    # Get the resulting base spectrum
    non_linear_spectrum = inter_array(non_linear_alphas * 100, LEDs)

    # Plot if needed
    if show_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(E_ref["Wavelength (nm)"], E_ref["Irradiance (W/m^2/nm)"], label="AM1.5G", color="black")
        plt.plot(non_linear_spectrum["Wavelength (nm)"], non_linear_spectrum["Irradiance (W/m^2/nm)"],
                 label="Base Spectrum", color="orange")
        plt.xlim(beginning_w, end_w)
        plt.ylim(bottom=0)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Irradiance (W/m²/nm)")
        plt.title("Comparison of Target Spectrum and Base Spectrum")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return non_linear_spectrum, non_linear_alphas

def get_LEDs(beginning_w,end_w,Solar_Simulator_LEDs_path):

    #LEDs is a dictionnary storing for each key "LED1", "LED2", ... a list of the form [power,df]: [[10, df], [20,df],...,[100,df]]
    #Where df is a 2 columns spectrum dataframe containing the wavelength and irradiance of the LED at the corresponding power 
    LEDs = {}
    LED_pathlist = [f for f in os.listdir(Solar_Simulator_LEDs_path) if f != '.DS_Store']

    for LED_folder in sorted(LED_pathlist, key=lambda x: int(x[3:])):
        LED_i = []
        for LED_file in sorted(os.listdir(Solar_Simulator_LEDs_path+"/"+LED_folder)):
            
            power = int(LED_file.split("_")[1])
            files = os.listdir(Solar_Simulator_LEDs_path+"/"+LED_folder+"/"+LED_file)
            f = list(filter(lambda x: x.startswith("Spectrum"),files))[0]
            
            df = pd.read_csv(Solar_Simulator_LEDs_path+"/"+LED_folder+"/"+LED_file+"/"+f, delimiter=" ", skiprows=1)
            df.columns = ['Wavelength (nm)', 'Irradiance (W/m^2/nm)']
            #conversion in the unit W/m²/nm
            df['Irradiance (W/m^2/nm)']=df['Irradiance (W/m^2/nm)'].div(100)
            #beginning_w = beginning wavelength
            #end_w = ending wavelength
            df = df[df['Wavelength (nm)']<end_w]
            df = df[df['Wavelength (nm)']>beginning_w]

            LED_i.append([power,df])
        
        LED_i = sorted(LED_i, key=lambda x: x[0])
    
        LEDs[LED_folder] = LED_i
    return LEDs

def get_nonlinear_recipe(beginning_w, end_w, E_ref, LEDs,initial_guess=None):
    data = E_ref[(E_ref['Wavelength (nm)']>beginning_w) & (E_ref['Wavelength (nm)']<end_w)].copy()
    Target = data['Irradiance (W/m^2/nm)'].values.reshape(-1, 1)
    
    #Objective function to minimize
    def nonlinear_objective_function(alphas):
        # Define the objective function to minimize
        functions = inter_array(np.array(alphas)*100,LEDs)

        #Putting all the data on the same wavelength grid
        interpolate_wavelength = pd.merge_asof(data, functions, left_on='Wavelength (nm)', right_on='Wavelength (nm)', direction='nearest')

        #Compute the norm of the difference 
        difference = Target - interpolate_wavelength['Irradiance (W/m^2/nm)_y'].values.reshape(-1, 1)
        norm = np.linalg.norm(difference)/np.linalg.norm(Target)
        return norm

    # Define the bounds and intial condition for optimization
    bounds = [(0, 1)] * 21


    if initial_guess == None:
        initial_guess = get_linear_recipe(beginning_w, end_w, data, LEDs)

    result = minimize(nonlinear_objective_function, initial_guess, method='Nelder-Mead', bounds=bounds)
    corrected_alphas = result.x

    return corrected_alphas

def inter_array(power_coefficients, LEDs):

    """
    Given the power coefficients {α} (power_coefficients), we calculate the cumulative spectral irradiance from multiple LED sources, interpolating for each LED.    irradiance contributions from multiple LEDs, each scaled by its respective power coefficient.
    The results are interpolated onto a common wavelength grid for consistent comparison.  
    """
    #call the inter function for each LED and sums the results

    # Define a common wavelength grid based on the first LED's wavelength range
    common_wavelengths = LEDs["LED1"][0][1]['Wavelength (nm)'].values
    
    # Initialize the cumulative irradiance array
    cumulative_irradiance = np.zeros_like(common_wavelengths, dtype=float)
    
    for i in range(1, 22):
        spec_i = inter(i, power_coefficients[i-1], LEDs)
        
        # Interpolate spec_i to the common wavelength grid
        interpolated_irradiance = np.interp(common_wavelengths, spec_i['Wavelength (nm)'].values, spec_i['Irradiance (W/m^2/nm)'].values)
        
        # Add the interpolated irradiance to the cumulative irradiance
        cumulative_irradiance += interpolated_irradiance
    
    # Create the resulting DataFrame
    result = pd.DataFrame({
        'Wavelength (nm)': common_wavelengths,
        'Irradiance (W/m^2/nm)': cumulative_irradiance
    })

    return result

def inter(LED_number, power_ratio,LEDs):
    """
    Does a linear interpolation of α between the two closes 10% interval of {α}. E.g. α = 25% will interpolate between LED at 20% and 30%.
    """
    #interpolate a LED spectrum to any power ratio
    if power_ratio%10 == 0:
        if power_ratio == 0:
            w = LEDs[f"LED{LED_number}"][int(power_ratio/10)-1][1].copy()
            w['Irradiance (W/m^2/nm)'] = 0
            return w
        return LEDs[f"LED{LED_number}"][int(power_ratio/10)-1][1].copy()
    else:
        index_min_power = int(power_ratio/10)-1
        index_max_power = index_min_power+1
        LED = f"LED{LED_number}"
        LED_i = LEDs[LED]
        if index_min_power == -1:
            max_irradiance = LED_i[0][1]
            min_irradiance = max_irradiance.copy()
            min_irradiance['Irradiance (W/m^2/nm)'] = 0
        else :
            
            min_irradiance = LED_i[index_min_power][1]
            max_irradiance = LED_i[min(index_max_power,9)][1]
        
        spectrum = min_irradiance.copy()

        #Interpolation between the two closest measured powers
        spectrum['Irradiance (W/m^2/nm)'] = min_irradiance['Irradiance (W/m^2/nm)'] + (power_ratio/10-(index_min_power+1))*(max_irradiance['Irradiance (W/m^2/nm)']-min_irradiance['Irradiance (W/m^2/nm)'])
        return spectrum

def get_linear_recipe(beginning_w, end_w, E_ref, LEDs):
    """
    Get {α} from gram-schmidt orthogonalization.
    """
    data = E_ref[(E_ref['Wavelength (nm)']>beginning_w) & (E_ref['Wavelength (nm)']<end_w)].copy()

    #Creation of the non orthonormal basis
    led_basis = []
    for led_data in LEDs.values():
        df = led_data[-1][1]
        merged_df = pd.merge_asof(data, df, on='Wavelength (nm)', direction='nearest')
        merged_df = pd.merge_asof(data, df, on='Wavelength (nm)', direction='nearest')
        led_basis.append(merged_df['Irradiance (W/m^2/nm)_y'].values)
    led_basis = np.array(led_basis)

    #create an orthonomral basis from this basis "ortho" and a change of basis matrix "change" (matrix A)
    ortho, change = gram_schmidt(led_basis)

    #Projection of the sun spectrum on the orthonormal basis
    mus = np.dot(data['Irradiance (W/m^2/nm)'], ortho.T)

    #Exact result of linear optimization on R (real numbers)
    alphas = np.dot(change.T, mus)
    #print('GS- alphas unbounded:')
    #print(alphas)
    #Adding constraints to the alphas: 0<=alphas<=1
    alphas_bounded = [max(0,min(1, x)) for x in alphas]

    #Target = data['Irradiance (W/m^2/nm)'].values.reshape(-1, 1)

    # Define the objective function to minimize
    #def linear_objective_function(alphas):
    #    difference = Target - np.sum(np.multiply(alphas, led_basis.T), axis=1).reshape(-1, 1)
    #    norm = np.linalg.norm(difference)/np.linalg.norm(Target)
    #    return norm
    
    #Defining the bounds for the alphas and the initial guess
    #bounds = [(0, 1)] * led_basis.shape[0]
    #initial_guess = alphas_bounded

    #result = minimize(linear_objective_function, initial_guess, method='Nelder-Mead', bounds=bounds)
    #optimized_alphas = result.x
   # print('GS -alphas bounded:')
    #print(alphas_bounded)

    return alphas_bounded

def gram_schmidt(basis):
    num_leds = basis.shape[0]
    orthonormal_basis = np.zeros_like(basis)
    change_of_base_matrix = np.zeros((num_leds, num_leds))
    norms = np.zeros(num_leds)
    for i in range(num_leds):
        
        projection = np.dot(basis[i], orthonormal_basis[:i].T)
        projection = projection.reshape(-1, 1)  # Reshape projection array
        orthogonal_vector = basis[i] - np.sum(projection * orthonormal_basis[:i], axis=0)
        orthonormal_basis[i] = orthogonal_vector / np.linalg.norm(orthogonal_vector)
        norms[i] = np.linalg.norm(orthogonal_vector)
        for j in range(i):
            for k in range(j,i):
                
                coefficient = np.dot(basis[i], orthonormal_basis[k])*change_of_base_matrix[k,j]/norms[i]
                change_of_base_matrix[i,j] -= coefficient

        change_of_base_matrix[i, i] = 1/norms[i]
    
    return orthonormal_basis, change_of_base_matrix

def get_inverse_coeffs(LEDs):
    """
    gets the 5 polynomial coefficients which fits the inverse relationship between alpah_real and alpha theoretical

    Parameters:
    LEDs (dict): A dictionary containing the LED spectra data.

    Returns:
    list: A list of polynomial coefficients for each LED, representing the inverse relationship between irradiance and power.
    """
    LEDs_irradiances = {}
    for LED in LEDs:
        coefficients = []
        for power, df in LEDs[LED]:
            irradiance = np.trapezoid(df['Irradiance (W/m^2/nm)'],x=df['Wavelength (nm)'])
            coefficients.append([power,irradiance])
        LEDs_irradiances[LED] = coefficients

    LEDs_coefficients = {}
    for LED in LEDs_irradiances:
        powers = [0]
        coefficients = [0]
        diff = [0]
        for power, irradiance in LEDs_irradiances[LED]:
            powers.append(power)
            coefficients.append(irradiance/LEDs_irradiances[LED][-1][1]*100)
            diff.append(irradiance/LEDs_irradiances[LED][-1][1]*100-power)
        LEDs_coefficients[LED] = [powers, coefficients]
    
    inverse_coeffs = []
    for LED in LEDs_coefficients:
        poly_power = 5
        inverse_coeffs.append(np.polyfit(LEDs_coefficients[LED][1], LEDs_coefficients[LED][0], poly_power))
        
    return inverse_coeffs

def get_theo_coeff(power_coefficients,LEDs):
    """
    Calculates the theoretical coefficients for each LED based on the given power coefficients.
    Is used for? - gets the coefficient if alpha was linear - that is change in intensity is linear with change in power setting
    
    Parameters:
    power_coefficients (list): A list of power coefficients for each LED.
    LEDs (dict): A dictionary containing the LED spectra data.

    Returns:
    numpy.ndarray: An array of theoretical coefficients as percentages.
    """

    theo_coeffs = []
    for i in range(1,22):
        spec = inter(i, power_coefficients[i-1],LEDs)
        max_spec = inter(i, 100,LEDs)
        theo_coeffs.append(np.trapezoid(spec['Irradiance (W/m^2/nm)'],x=spec['Wavelength (nm)'])/np.trapezoid(max_spec['Irradiance (W/m^2/nm)'],x=max_spec['Wavelength (nm)']))
    return np.array(theo_coeffs)*100

def split_spectrum(power_coefficients, n1, LEDs):
    power_coefs1 = power_coefficients.copy()
    power_coefs2 = power_coefficients.copy()
    for i in range(21):
        if i < n1:
            power_coefs2[i] = 0
        else:
            power_coefs1[i] = 0
    return inter_array(power_coefs1, LEDs), inter_array(power_coefs2, LEDs)

#Computing new mismatch factor and spectrum after calibration
def meusel_calibration(s1,s2, n, inverse_coeffs, theo_coeffs, LEDs, topcell, bottomcell, E_ref):
    #print(f"{n}:")
    A1,A2 = solve_Meusel_system(topcell,bottomcell,s1,s2,E_ref)
    A1 = max(0.0,A1); A2 = max(0.0,A2)
    #print(f"A1 = {A1:.3f}, A2 = {A2:.3f}")
    new_coeffs = []
    for i in range(len(theo_coeffs)):
        if i < n:
            new_coeffs.append(max(0,min(100.0,np.polyval(inverse_coeffs[i],theo_coeffs[i]*A1))))
        else:
            new_coeffs.append(max(0,min(100.0,np.polyval(inverse_coeffs[i],theo_coeffs[i]*A2))))

    #print(f"{[str(coefficients[i])+':'+str(new_coeffs[i]) for i in range(21)]}")
    new_spectrum = inter_array(new_coeffs, LEDs)
    new_coeffs_s1 = np.copy(new_coeffs) 
    new_coeffs_s1[n:] = 0
    new_coeffs_s2 = np.copy(new_coeffs) 
    new_coeffs_s2[:n] = 0

    # new split spectrum
    new_s1 = inter_array(new_coeffs_s1, LEDs)
    new_s2 = inter_array(new_coeffs_s2, LEDs)
    return new_spectrum, new_coeffs,new_s1,new_s2

#Solving calibration system of equations
def solve_Meusel_system(topcell, bottomcell, s1, s2, E_ref):

    # Drop EQE
    #topcell.drop('EQE (%)', axis=1, inplace=True)
    #bottomcell.drop('EQE (%)', axis=1, inplace=True)

    # Rename s1, s2, top, bottom to _TOP, _BOT
    topcell = topcell.rename(columns={"SR (A/W)": "SR_TOP (A/W)"})
    bottomcell = bottomcell.rename(columns={"SR (A/W)": "SR_BOT (A/W)"})

    #rename Irradiance (W/m^2/nm) to ref and s1 and s2
    E_ref = E_ref.rename(columns={"Irradiance (W/m^2/nm)":"Irradiance_ref"})
    #print("E_ref")
    #print(E_ref.head())
    s1 = s1.rename(columns={"Irradiance (W/m^2/nm)":"Irradiance_s1"})
    #print("s1")
    #print(s1.head())
    s2 = s2.rename(columns={"Irradiance (W/m^2/nm)":"Irradiance_s2"})
    #print("s2")
    #print(s2.head())

    #print("topcell")
    #print(topcell.head())
    #print("bottomcell")
    #print(bottomcell.head())

    #merge the EQE values and the spectra
    df = pd.merge_asof(E_ref, s1, left_on='Wavelength (nm)', right_on='Wavelength (nm)', direction='nearest')

    df = pd.merge_asof(df, s2, left_on='Wavelength (nm)', right_on='Wavelength (nm)', direction='nearest')

    df = pd.merge_asof(df, topcell, left_on='Wavelength (nm)', right_on='Wavelength (nm)', direction='nearest')
 
    df = pd.merge_asof(df, bottomcell, left_on='Wavelength (nm)', right_on='Wavelength (nm)', direction='nearest')

    #print(df.head())

    LHS = np.zeros((2, 2))
    LHS[0,0] = np.trapezoid(df["SR_TOP (A/W)"]*df["Irradiance_s1"], df["Wavelength (nm)"])
    LHS[0,1] = np.trapezoid(df["SR_TOP (A/W)"]*df["Irradiance_s2"], df["Wavelength (nm)"])
    LHS[1,0] = np.trapezoid(df["SR_BOT (A/W)"]*df["Irradiance_s1"], df["Wavelength (nm)"])
    LHS[1,1] = np.trapezoid(df["SR_BOT (A/W)"]*df["Irradiance_s2"], df["Wavelength (nm)"])

    RHS = np.zeros((2,1))
    RHS[0] = np.trapezoid(df["SR_TOP (A/W)"]*df["Irradiance_ref"], df["Wavelength (nm)"])
    RHS[1] = np.trapezoid(df["SR_BOT (A/W)"]*df["Irradiance_ref"], df["Wavelength (nm)"])

    sol = np.linalg.solve(LHS,RHS)
    A1 = sol[0][0]; A2 = sol[1][0]
    return A1,A2

# This code chooses the best number of LEDs for the calibration
def calibrate_based_on_best_splitting(DUT_cells, RCs, E_ref, base_alphas,recipe_name, Solar_Simulator_LEDs_path, beginning_w=300, end_w=1200):
    """
    Function to optimize mismatch factors by calibrating the spectrum.

    Parameters:
    DUT_cells (dict): Dictionary of device under test (DUT) sub-cells, e.g., {'TOP': PVSK, 'BOT': Si}
    RCs (dict): Dictionary of reference cells, e.g., {'KG3': KG3, 'BL7': BL7}
    E_ref (DataFrame): Reference spectrum data, i.e. AM15G.
    recipe_name (str): Name of the recipe to be generated.
    spectrum_number (int): Identifier for the spectrum recipe.
    LEDs (DataFrame): LED data to be used for optimization.
    beginning_w (float, optional): Starting wavelength for the spectrum calibration.
    end_w (float, optional): Ending wavelength for the spectrum calibration.
    
    Returns:
    dict: Contains the best number of LEDs, mismatch factors, and the calibrated recipe.
    """
    LEDs = get_LEDs(beginning_w, end_w, Solar_Simulator_LEDs_path)
    
    # Error check: ensure both dictionaries have the same length
    if len(DUT_cells) != len(RCs):
        raise ValueError("DUT_cells and RCs must have the same number of entries.")

    # Check that the maximum EQE of each Ref and DUT pair aligns within 300nm
    for (dut_name, dut_cell), (ref_name, ref_cell) in zip(DUT_cells.items(), RCs.items()):
        # Find the wavelength of maximum EQE for each DUT and Ref cell
        max_eqe_dut_wavelength = dut_cell['Wavelength (nm)'][dut_cell['EQE (%)'].idxmax()]
        max_eqe_ref_wavelength = ref_cell['Wavelength (nm)'][ref_cell['EQE (%)'].idxmax()]

        # Check if the difference in wavelengths is more than 300nm
        if abs(max_eqe_dut_wavelength - max_eqe_ref_wavelength) > 300:
            warnings.warn(f"Warning: The reference cell '{ref_name}' and DUT cell '{dut_name}' EQE peaks are mismatched "
                          f"(difference of {abs(max_eqe_dut_wavelength - max_eqe_ref_wavelength)}nm). "
                          f"Please confirm the correct pairing.")
    
    base_spectrum = inter_array(base_alphas, LEDs)

    if beginning_w and end_w:
        base_spectrum = base_spectrum[(base_spectrum['Wavelength (nm)'] >= beginning_w) &
                                                (base_spectrum['Wavelength (nm)'] <= end_w)]

    # Calculate the mismatch factors before calibration
    first_dut, first_ref = list(DUT_cells.items())[0], list(RCs.items())[0]
    
    Mtop_before = get_mismatch_factor(E_ref, base_spectrum, first_ref[1], first_dut[1])
    
    second_dut, second_ref = list(DUT_cells.items())[1], list(RCs.items())[1]
    Mbot_before = get_mismatch_factor(E_ref, base_spectrum, second_ref[1], second_dut[1])
    
    print(f"Mismatch factors before calibration: Mtop = {Mtop_before:.3f}, Mbot = {Mbot_before:.3f}")

    # Initialize arrays to store mismatch factor results
    X = np.arange(1, 21, 1)  # Number of LEDs in the first source
    Ytop, Ybot, Ymean = [], [], []

    # Get inverse and theoretical coefficients
    inverse_coeffs = get_inverse_coeffs(LEDs)
    theo_coeffs = get_theo_coeff(base_alphas, LEDs) 

    # Loop over LED numbers and calculate mismatch factors for different splits
    for n in X:
        s1, s2 = split_spectrum(base_alphas, n, LEDs)

        # Calibrate the spectrum and return the new alphas after calibration
        spectrum_calibrated, _, _, _ = meusel_calibration(s1, s2, n, inverse_coeffs, theo_coeffs, LEDs, first_dut[1], second_dut[1], E_ref)
        
        # Calculate mismatch factors for top and bottom cells
        Mtop = get_mismatch_factor(E_ref, spectrum_calibrated, first_ref[1], first_dut[1])
        Mbot = get_mismatch_factor(E_ref, spectrum_calibrated, second_ref[1], second_dut[1])
        
        # Store results
        Ytop.append(abs(1 - Mtop))
        Ybot.append(abs(1 - Mbot))
        Ymean.append((abs(1 - Mtop) + abs(1 - Mbot)) / 2)

    # Find the best number of LEDs for minimizing mismatch factors
    best = X[np.argmin(Ymean)]

    # Perform final calibration with the best LED split
    s1, s2 = split_spectrum(base_alphas, best, LEDs)
    spectrum_calibrated, calibrated_alphas, new_S1, new_S2= meusel_calibration(s1, s2, best, inverse_coeffs, theo_coeffs, LEDs, first_dut[1], second_dut[1], E_ref)

    # Output the best mismatch factors
    Mtop_best = get_mismatch_factor(E_ref, spectrum_calibrated, first_ref[1], first_dut[1])
    Mbot_best = get_mismatch_factor(E_ref, spectrum_calibrated, second_ref[1], second_dut[1])
    print(f"Best calibration with {best} LEDs in source 1, Mtop = {Mtop_best:.3f}, Mbot = {Mbot_best:.3f}")
    
    # Return results as a dictionary
    return {
        "best_led_number": best,
        "Mtop_best": Mtop_best,
        "Mbot_best": Mbot_best,
        "calibrated_alphas": calibrated_alphas,
        "calibrated_spectrum": spectrum_calibrated,
        "split_calibrated_spectrum":[s1, s2]
        }

def read_wspc_recipe(path):
    with open(path,"r") as file:
        data = json.load(file)
    
    recipe = []
    for i in range(1,22):
        recipe.append(data['Settings']['SubSettings']['Channels']['SubSettings']['Channel'+str(i)]['Value'])
    
    return np.array(recipe)


# Calculating the Mismatch factors
def get_mismatch_factor(E_ref, E_sim, RC, DUT):
    """
    Calculates the mismatch factor between reference and simulated spectra for given reference and DUT cells.

    Parameters:
    E_ref (pandas.DataFrame): DataFrame containing the reference spectrum with 'Wavelength (nm)' and 'Irradiance (W/m^2/nm)' columns.
    E_sim (pandas.DataFrame): DataFrame containing the simulated spectrum with 'Wavelength (nm)' and 'Irradiance (W/m^2/nm)' columns.
    RC (pandas.DataFrame): DataFrame containing the reference cell's spectral response with 'Wavelength (nm)' and 'SR' columns.
    DUT (pandas.DataFrame): DataFrame containing the DUT cell's spectral response with 'Wavelength (nm)' and 'SR' columns.

    Returns:
    float: The calculated mismatch factor.
    """
    # Rename the columns in the original dataframes
    E_ref = E_ref.rename(columns={"Irradiance (W/m^2/nm)": "Irradiance_ref"})
    E_sim = E_sim.rename(columns={"Irradiance (W/m^2/nm)": "Irradiance_sim"})
    RC = RC.rename(columns={"SR (A/W)": "SR_RC (A/W)"})
    DUT = DUT.rename(columns={"SR (A/W)": "SR_DUT (A/W)"})

    # Merge the dataframes on 'Wavelength (nm)'
    df = pd.merge_asof(E_ref, E_sim, left_on='Wavelength (nm)', right_on='Wavelength (nm)', direction='nearest')
    df = pd.merge_asof(df, RC, left_on='Wavelength (nm)', right_on='Wavelength (nm)', direction='nearest')
    df = pd.merge_asof(df, DUT, left_on='Wavelength (nm)', right_on='Wavelength (nm)', direction='nearest')

    # Calculate the mismatch factor
    MM = (np.trapezoid(df["SR_DUT (A/W)"] * df["Irradiance_sim"], df["Wavelength (nm)"]) *
          np.trapezoid(df["SR_RC (A/W)"] * df["Irradiance_ref"], df["Wavelength (nm)"]) /
          np.trapezoid(df["SR_RC (A/W)"] * df["Irradiance_sim"], df["Wavelength (nm)"]) /
          np.trapezoid(df["SR_DUT (A/W)"] * df["Irradiance_ref"], df["Wavelength (nm)"]))

    return MM


def calibrate_based_on_specific_splitting(DUT_cells, RCs, E_ref, base_alphas, k, recipe_name, Solar_Simulator_LEDs_path, beginning_w=300, end_w=1200):
    """
    Function to optimize mismatch factors by calibrating the spectrum.

    Parameters:
    DUT_cells (dict): Dictionary of device under test (DUT) sub-cells, e.g., {'TOP': PVSK, 'BOT': Si}
    RCs (dict): Dictionary of reference cells, e.g., {'KG3': KG3, 'BL7': BL7}
    E_ref (DataFrame): Reference spectrum data, i.e. AM15G.
    recipe_name (str): Name of the recipe to be generated.
    spectrum_number (int): Identifier for the spectrum recipe.
    LEDs (DataFrame): LED data to be used for optimization.
    beginning_w (float, optional): Starting wavelength for the spectrum calibration.
    end_w (float, optional): Ending wavelength for the spectrum calibration.
    
    Returns:
    dict: Contains the best number of LEDs, mismatch factors, and the calibrated recipe.
    """
    LEDs = get_LEDs(beginning_w, end_w, Solar_Simulator_LEDs_path)
    
    # Error check: ensure both dictionaries have the same length
    if len(DUT_cells) != len(RCs):
        raise ValueError("DUT_cells and RCs must have the same number of entries.")

    # Check that the maximum EQE of each Ref and DUT pair aligns within 300nm
    for (dut_name, dut_cell), (ref_name, ref_cell) in zip(DUT_cells.items(), RCs.items()):
        print("dut_cell")
        print(dut_cell.head())
        # Find the wavelength of maximum EQE for each DUT and Ref cell
        max_eqe_dut_wavelength = dut_cell['Wavelength (nm)'][dut_cell['EQE (%)'].idxmax()]
        max_eqe_ref_wavelength = ref_cell['Wavelength (nm)'][ref_cell['EQE (%)'].idxmax()]

        # Check if the difference in wavelengths is more than 300nm
        if abs(max_eqe_dut_wavelength - max_eqe_ref_wavelength) > 300:
            warnings.warn(f"Warning: The reference cell '{ref_name}' and DUT cell '{dut_name}' EQE peaks are mismatched "
                          f"(difference of {abs(max_eqe_dut_wavelength - max_eqe_ref_wavelength)}nm). "
                          f"Please confirm the correct pairing.")
    
    # Import optimized recipe
    #optimized_alphas = read_wspc_recipe(f"./Generated recipes/Scaled_recipe/Meusel/1_20_calibrated.wspc")
    # optimized_alphas = read_wspc_recipe(r"Generated recipes/1_19_Gram1.wspc")
    # Create the initial spectrum dataframe from the optimized recipe
    base_spectrum = inter_array(base_alphas, LEDs)
    #print("optimized_spectrum")
    #print(optimized_spectrum.head())
    #print("E_ref")
    #print(E_ref.head())
    # Optional range of wavelengths for calibration

    if beginning_w and end_w:
        base_spectrum = base_spectrum[(base_spectrum['Wavelength (nm)'] >= beginning_w) &
                                                (base_spectrum['Wavelength (nm)'] <= end_w)]

    # Calculate the mismatch factors before calibration
    first_dut, first_ref = list(DUT_cells.items())[0], list(RCs.items())[0]
    
    Mtop_before = get_mismatch_factor(E_ref, base_spectrum, first_ref[1], first_dut[1])
    
    second_dut, second_ref = list(DUT_cells.items())[1], list(RCs.items())[1]
    Mbot_before = get_mismatch_factor(E_ref, base_spectrum, second_ref[1], second_dut[1])
    
    print(f"Mismatch factors before calibration: Mtop = {Mtop_before:.3f}, Mbot = {Mbot_before:.3f}")

    # Initialize arrays to store mismatch factor results
    X = np.arange(1, 21, 1)  # Number of LEDs in the first source
    Ytop, Ybot, Ymean = [], [], []

    # Get inverse and theoretical coefficients
    inverse_coeffs = get_inverse_coeffs(LEDs)
    theo_coeffs = get_theo_coeff(base_alphas, LEDs) 

    # Perform final calibration with the best LED split
    s1_base, s2_base = split_spectrum(base_alphas, k, LEDs)
    spectrum_calibrated, calibrated_alphas, new_S1, new_S2= meusel_calibration(s1_base, s2_base, k, inverse_coeffs, theo_coeffs, LEDs, first_dut[1], second_dut[1], E_ref)
    s1_cal, s2_cal = split_spectrum(calibrated_alphas, k, LEDs)

    # Output the best mismatch factors
    Mtop = get_mismatch_factor(E_ref, spectrum_calibrated, first_ref[1], first_dut[1])
    Mbot = get_mismatch_factor(E_ref, spectrum_calibrated, second_ref[1], second_dut[1])
    print(f"Calibration with {k} LEDs in source 1, Mtop = {Mtop:.3f}, Mbot = {Mbot:.3f}")

    
    # Return results as a dictionary
    return {
        "k": k,
        "Mtop": Mtop,
        "Mbot": Mbot,
        "calibrated_alphas": calibrated_alphas,
        "calibrated_spectrum": spectrum_calibrated,
        "split_base_spectrum":[s1_base, s2_base],
        "split_calibrated_spectrum":[s1_cal, s2_cal]
        }