# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    LinearRegression.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: yhwang <yhwang@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/26 23:57:24 by yhwang            #+#    #+#              #
#    Updated: 2024/06/27 00:04:10 by yhwang           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

class LinearRegression:
        def __init__(self, theta1 = 0, theta2 = 0):
                self.theta1 = theta1
                self.theta2 = theta2
                
        def get_theta(self):
                print("theta1 = ", self.theta1, ", theta2 = ", self.theta2)
                return self.theta1, self.theta2
