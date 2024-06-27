# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: yhwang <yhwang@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/26 23:15:10 by yhwang            #+#    #+#              #
#    Updated: 2024/06/27 01:42:42 by yhwang           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

def main():
        # data = pd.read_csv('data.csv')

        # plt.figure(figsize=(10, 6))
        # plt.scatter(data['km'], data['price'], color='blue', alpha=0.5)
        # plt.title('Mileage vs. Price')
        # plt.xlabel('Mileage')
        # plt.ylabel('Price')
        # plt.grid(True)
        # plt.show(block=True)
        
        # lr = LinearRegression()
        
        # lr.get_theta()
        
        # lr.theta1 = 10
        # lr.theta2 = 20
        
        # lr.get_theta()


        data = pd.read_csv('data.csv')

        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(data['km'], data['price'], color='blue', alpha=0.5)
        ax.set_title('Mileage vs. Price')
        ax.set_xlabel('Mileage')
        ax.set_ylabel('Price')
        ax.grid(True)
        plt.show()

        lr = LinearRegression()

        print("Initial theta values:")
        lr.get_theta()

        lr.theta1 = 10
        lr.theta2 = 20

        print("Updated theta values:")
        lr.get_theta()

        # Keep the plot open
        input("Press Enter to close the plot...")

if __name__ == "__main__":
        main()
