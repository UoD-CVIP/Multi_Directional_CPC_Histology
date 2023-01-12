import numpy
import matplotlib
import matplotlib.pyplot as plt


SDNM_loss = [5.257608, 0.989027, 1.349410, 0.752091, 0.523022, 0.529297, 0.645220, 0.539003, 0.436536, 0.406119,
             0.374055, 0.350006, 0.091212, 0.145850, 0.685074, 0.119619, 0.090556, 0.039747, 0.098223, 0.058772]
SDNM_val  = [0.143841, 0.504554, 0.774139, 0.373294, 0.188954, 0.447829, 0.605157, 0.451862, 0.296148, 0.233436,
             0.419020, 0.259416, 0.092819, 0.039086, 0.286215, 0.008089, 0.006664, 0.030424, 0.008190, 0.051909]

fig, ax = plt.subplots()
ax.plot(list(range(20)), SDNM_loss)

plt.xticks(list(range(20)))
plt.yticks(list(numpy.arange(0., 5.25, 0.25)))

ax.set(xlabel="Epoch", ylabel="Loss")
ax.grid()


fig.savefig("test.png")
