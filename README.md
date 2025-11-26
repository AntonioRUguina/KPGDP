# KPGDP
Code for KPGDP Grasp Algorithm and MCM Algorithm

# Use

You can use the Main.py to execute the metaheuristic and matheuristic algorithm, that are implemented in Algorithm.py and PackingFast.py respectively. 

You can use the MainModels.py to execute the different models. F1, F2 and F3 can be executed with run_algorithm(), while p-chained can be executed with run_algorithm_chained()

# Output

In output folder all the results of the experiments are found, where the first column is the instance, the second is
the Algorithm used, the third is the time and the last one is the objective function reached. For Packaging algorithms, k solutions will be returned,
being the lowest the objective function reached by the method. For example:

GKD-b_41_n150_2_20.txt PakingBinary 29.311007499694824: 177.6

# Creation of RUMG Instances

In order to analyze the performance of commercial solvers, specifically Gurobi 11.0.1, and the algorithm developed in the previous section on realistic personnel assignment problems, a series of instances have been generated. These instances represent companies of various sizes, where the distance between employees is defined by the difference in certain numerical characteristics of interest, numbered as follows:

Gender: Gender of the employee. Possible values: {0, 1}.

Age: Age of the employee. All employees are older than 20 years.

Education Level: Takes values from 1 to 5.

Years of Experience: Number of years of work experience. Takes positive integer values.

Years in the Company: Number of years the employee has been with the company. This value is less than or equal to the years of experience.

Rank: Position rank within the company (e.g., junior, senior, director, etc.). Takes values from 1 to 6.

Department: Department to which the employee belongs. Categorical variable.


Therefore, the distance between two individuals is calculated as follows. For age, the absolute difference is taken. For department, a distance matrix is generated among departments ranging from 1 to 5, and the values of this matrix is multiplied by 5. The matrix represents the differences between departments because it is understood that similarities between departments can vary. For example, there are more similarities between two technical departments than between a technical department and a functional one. Finally, for the remaining categories, the absolute difference between values is taken and multiplied by 5.


The result is the RUMG instances summarized 
