# KPGDP
Code for KPGDP Grasp Algorithm and MCM Algorithm

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
