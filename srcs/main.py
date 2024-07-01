# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: yhwang <yhwang@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/26 23:15:10 by yhwang            #+#    #+#              #
#    Updated: 2024/07/02 00:06:50 by yhwang           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
import sys

g_flag_sigquit = False

def window_close(event):
        print("\nPlot closed")
        plt.close('all')
        sys.exit(0)

def main():
        data = pd.read_csv('data.csv')

        plt.ion() # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(data['km'], data['price'], color='blue', alpha=0.5)
        ax.set_title('Mileage vs Price')
        ax.set_xlabel('Mileage')
        ax.set_ylabel('Price')
        ax.grid(True)
        plt.show()

        lr = LinearRegression()

        print("Initial theta values:")
        lr.get_theta()
        
        x_values = data['km']
        y_values = lr.theta1 * x_values + lr.theta2
        line, = ax.plot(x_values, y_values, color='red', linewidth=2)
        plt.draw()



        # lr.theta1 = -0.016
        # lr.theta2 = 8000

        # print("Updated theta values:")
        # lr.get_theta()

        # y_values = lr.theta1 * x_values + lr.theta2
        # line.set_ydata(y_values)
        # ax.plot(x_values, y_values, color='red', linewidth=2)
        # plt.draw()

        fig.canvas.mpl_connect('close_event', window_close)

        input("Press Enter to close the plot")
        plt.close(fig)

if __name__ == "__main__":
        main()
