import numpy as np
import matplotlib.pyplot as plt

def discretize_domain(left, right, element, n, type):

    element_type_map = {'linear': 1, 'quadratic': 2, 'cubic': 3}
    element_type = element_type_map[element]

    if n%element_type != 1 and element_type != 1:
        print("NUMBER OF NODES IS INCOMPATIBLE WITH ELEMENT TYPE")
        n += element_type - n % element_type + 1
        print("CHANGING NUMBER OF NODES TO: ", n)

    if type == 'uniform':
        x = np.linspace(left, right, n)
    elif type == 'random':
        x = np.random.uniform(left, right, n-2)
        x = np.concatenate(([left], x, [right]))
        x = np.sort(x)

    #Now create a 2D array with each entry containing element coordinates
    
    y = []

    for i in range(0, len(x) - element_type, element_type):
        element_coords = []
        for j in range(0, element_type + 1):
            element_coords.append(x[i+j])
        y.append(element_coords)
        
    y = np.array(y)

    return y


def compute_shape_functions(degree):

    isodomain = np.linspace(-1, 1, degree + 1)
    vandermonde = []

    for i in range(degree + 1):
        row = [isodomain[i] ** j for j in range(degree + 1)]
        vandermonde.append(row)

    vandermonde = np.array(vandermonde)

    shape_coefficients = []
    
    for i in range(degree + 1):
        e_k = np.zeros(degree + 1)
        e_k[i] = 1
        coeffs = np.linalg.solve(vandermonde, e_k)
        reversed_coeffs = coeffs[::-1]
        polynomial = np.poly1d(reversed_coeffs)
        shape_coefficients.append(polynomial)

    shape_derivative = [shape_function.deriv() for shape_function in shape_coefficients]

    return shape_coefficients, shape_derivative


def matrix_polynomial_integrate(mat_1, mat_2, transpose):

    poly = []
    if transpose == 1:
        for i in range(len(mat_1)):
            row = []
            for j in range(len(mat_2)):
                row.append(mat_1[i] * mat_2[j])
            poly.append(row)

        integrated_poly = [[p.integ() for p in row] for row in poly]

        output = [[p(1) - p(-1) for p in row] for row in integrated_poly]
        return np.array(output)
    else:
        print("FUNCTION NOT SUPPORTED")

def assemble(matrices):
    no_elements = len(matrices)
    nodes_per_element = len(matrices[0])
    no_nodes = no_elements * nodes_per_element - no_elements + 1

    if len(matrices[0][0]) == 1:
        rank = 1
    else:
        rank = 2

    global_k = []
    global_b = []
    for i in range(no_nodes):
        row = []
        for j in range(no_nodes):
            row.append(0)
        global_k.append(row)
        global_b.append(0)

    
    for i in range(no_elements):
        element_map = {}
        for j in range(1, nodes_per_element + 1):
            element_map[j] = (nodes_per_element - 1) * i + j
        
        element_i_matrix = matrices[i]
        for j in range(len(element_i_matrix)):

            if rank == 2:
                for k in range(len(element_i_matrix[j])):
                    j_new = element_map[j+1] - 1
                    k_new = element_map[k+1] - 1
                    global_k[j_new][k_new] += element_i_matrix[j][k]
            elif rank == 1:
                j_new = element_map[j+1] - 1
                global_b[j_new] += element_i_matrix[j][0]

    if rank == 2:
        return np.array(global_k)
    elif rank == 1:
        return np.array(global_b)



def apply_bconds(stiffness_matrix, rhs_vector, displacement_bconds):
    node_num = displacement_bconds[0] - 1
    prescribed_displacement = displacement_bconds[1]
    stiffness_matrix[node_num][:] = 0
    stiffness_matrix[node_num][node_num] = 1
    rhs_vector[node_num] = prescribed_displacement

    return stiffness_matrix, rhs_vector


def globalize_soln(matrix, disp, x, isB):

    for i in range(len(x)):
        for j in range(len(matrix)):
            if j == 0:
                combined_p = matrix[0] * disp[(len(x[0])-1) * i]
            else:
                combined_p += matrix[j] * disp[(len(x[0])-1) * i + j]

        position = np.linspace(x[i][0], x[i][-1], 50)
        epsilon = (2 * position - (x[i][-1] + x[i][0])) / (x[i][-1] - x[i][0])

        if i == 0:
            if isB:
                global_values = combined_p(epsilon) * 2 / (x[i][-1] - x[i][0])
            else:
                global_values = combined_p(epsilon)
            global_position = position
        else:
            if isB:
                elemental_values = combined_p(epsilon) * 2 / (x[i][-1] - x[i][0])
            else:
                elemental_values = combined_p(epsilon)
            
            global_values = np.concatenate((global_values, elemental_values))
            global_position = np.concatenate((global_position, position))

    
    return global_position, global_values
    


def main():

    E = 70000
    A = 1e-6

    input_string = input("PLEASE INPUT THE END COORDINATES: ")
    left_end, right_end = map(float, input_string.split())

    #Discretize the domain into finite elements
    x_coords = discretize_domain(left = left_end, right = right_end, element = 'cubic', n = 505, type = 'uniform')
    number_of_elements = len(x_coords)

    print("THE DISCRETIZED DOMAIN IS: ", x_coords)

    degree = len(x_coords[0]) - 1
    shape_data, derivative_shape_data = compute_shape_functions(degree) #solve a matrix problem via isoparametric mapping

    print("SHAPE FUNCTIONS", shape_data, '\n',
          "DERIVATIVE OF SHAPE FUNCTIONS", derivative_shape_data, '\n',
          "NUMBER OF ELEMENTS", number_of_elements)

    element_stiffness_matrices = []
    element_rhs_vectors = []

    for element in range(number_of_elements):
        k_e = matrix_polynomial_integrate(derivative_shape_data, derivative_shape_data, transpose=1)
        k_e *= E * A * 2 / (x_coords[element][-1] - x_coords[element][0])
        f_e = np.array([2]) #Distributed force in element (choosing a constant for now)
        b_e = matrix_polynomial_integrate(shape_data, f_e, transpose=1)
        b_e *= (x_coords[element][-1] - x_coords[element][0]) / 2
        element_stiffness_matrices.append(k_e)
        element_rhs_vectors.append(b_e)

    element_stiffness_matrices = np.array(element_stiffness_matrices)
    element_rhs_vectors = np.array(element_rhs_vectors)

    K = assemble(element_stiffness_matrices)
    b = assemble(element_rhs_vectors)

    displacement_bconds = (0, 0)

    K, b = apply_bconds(K, b, displacement_bconds)

    a = np.linalg.solve(K, b)
    print("NODAL DISPLACEMENTS", a)


    global_position, global_displacements = globalize_soln(shape_data, a, x_coords, False)
    global_position, global_strains = globalize_soln(derivative_shape_data, a, x_coords, True)

    plt.figure()
    plt.plot(global_position, global_displacements)
    plt.title("Displacements")
    plt.xlabel('Position')
    plt.ylabel('Displacement')

    plt.figure()
    plt.plot(global_position, global_strains)
    plt.title("Strains")
    plt.xlabel("Position")
    plt.ylabel("Strains")


    plt.show()



main()