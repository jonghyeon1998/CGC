import jax
from jax import vmap
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

#Create Matern-2.5 kernel functions and derivatives



def kernel_u(s,t,params):
  K = -5*(s-t)*jnp.exp(-jnp.sqrt(5)*jnp.abs(s-t)/params[0])*(params[0]+jnp.sqrt(5)*jnp.abs(s-t))/(3*params[0]**3)
  return K

def kernel_z(s,t,params):
  K = 5*(s-t)*jnp.exp(-jnp.sqrt(5)*jnp.abs(s-t)/params[0])*(params[0]+jnp.sqrt(5)*jnp.abs(s-t))/(3*params[0]**3)
  return K

def kernel_uz(s,t,params):
  K = 5*jnp.exp(-jnp.sqrt(5)*jnp.abs(s-t)/params[0])*(jnp.sqrt(5)*params[0]*jnp.abs(s-t)+params[0]**2-5*(s-t)**2)/(3*params[0]**4)
  return K

def kernel_uuz(s,t,params):
  K = 25*(s-t)*jnp.exp(-jnp.sqrt(5)*jnp.abs(s-t)/params[0])*(jnp.sqrt(5)*jnp.abs(s-t)-3*params[0])/(3*params[0]**5)
  return K

def kernel_uzz(s,t,params):
  K = -25*(s-t)*jnp.exp(-jnp.sqrt(5)*jnp.abs(s-t)/params[0])*(jnp.sqrt(5)*jnp.abs(s-t)-3*params[0])/(3*params[0]**5)
  return K

def kernel_uu(s,t,params):
  K =  -5*jnp.exp(-jnp.sqrt(5)*jnp.abs(s-t)/params[0])*(jnp.sqrt(5)*params[0]*jnp.abs(s-t)+params[0]**2-5*(s-t)**2)/(3*params[0]**4)
  return K

def kernel_zz(s,t,params):
  K =  -5*jnp.exp(-jnp.sqrt(5)*jnp.abs(s-t)/params[0])*(jnp.sqrt(5)*params[0]*jnp.abs(s-t)+params[0]**2-5*(s-t)**2)/(3*params[0]**4)
  return K

def kernel_uuzz(s,t,params):
  K = -25*jnp.exp(-jnp.sqrt(5)*jnp.abs(s-t)/params[0])*(5*jnp.sqrt(5)*params[0]*jnp.abs(s-t)-(3*params[0]**2+5*(s-t)**2))/(3*params[0]**6)
  return K

#Kernel matrix K
def K_Matrix(X,Y,params,reg=False,nugget=10**-5):

  size=len(X[:,0])
  size2=len(Y[:,0])
  X0=jnp.transpose(jnp.tile(X[:,0],(size2,1)))
  Y0=jnp.transpose(jnp.tile(Y[:,0],(size,1)))

  val = vmap(lambda s, t: kernel(s,t,params))(X0.flatten(),np.transpose(Y0).flatten())

  K_matrix=np.reshape(val,(size,size2))
  if reg==True and size==size2:
    K_matrix+=+nugget*jnp.eye(size)
  return K_matrix

#First derivative with respect to u
def K_du(X,Y,params,reg=False,nugget=10**-5):
  size=len(X[:,0])
  size2=len(Y[:,0])
  X0=jnp.transpose(jnp.tile(X[:,0],(size2,1)))
  Y0=jnp.transpose(jnp.tile(Y[:,0],(size,1)))

  K_du = vmap(lambda s, t: kernel_u(s,t,params))(X0.flatten(),np.transpose(Y0).flatten())
  K_du = np.reshape(K_du,(size,size2))

  if reg==True and size==size2:
    K_du+=+nugget*jnp.eye(size)
  return K_du

#First derivative with respect to z
def K_dz(X,Y,params,reg=False,nugget=10**-5):
  size=len(X[:,0])
  size2=len(Y[:,0])
  X0=jnp.transpose(jnp.tile(X[:,0],(size2,1)))
  Y0=jnp.transpose(jnp.tile(Y[:,0],(size,1)))

  K_dz = vmap(lambda s, t: kernel_z(s,t,params))(X0.flatten(),np.transpose(Y0).flatten())
  K_dz = np.reshape(K_dz,(size,size2))

  if reg==True and size==size2:
    K_dz+=+nugget*jnp.eye(size)
  return K_dz

#Second derivative with respect to z
def K_dzz(X,Y,params,reg=False,nugget=10**-5):
  size=len(X[:,0])
  size2=len(Y[:,0])
  X0=jnp.transpose(jnp.tile(X[:,0],(size2,1)))
  Y0=jnp.transpose(jnp.tile(Y[:,0],(size,1)))

  K_dzz = vmap(lambda s, t: kernel_uu(s,t,params))(X0.flatten(),np.transpose(Y0).flatten())
  K_dzz = np.reshape(K_dzz,(size,size2))

  if reg==True and size==size2:
    K_dzz+=+nugget*jnp.eye(size)
  return K_dzz

#Second derivative with respect to u
def K_duu(X,Y,params,reg=False,nugget=10**-5):
  size=len(X[:,0])
  size2=len(Y[:,0])
  X0=jnp.transpose(jnp.tile(X[:,0],(size2,1)))
  Y0=jnp.transpose(jnp.tile(Y[:,0],(size,1)))

  K_duu = vmap(lambda s, t: kernel_uu(s,t,params))(X0.flatten(),np.transpose(Y0).flatten())
  K_duu = np.reshape(K_duu,(size,size2))

  if reg==True and size==size2:
    K_duu+=+nugget*jnp.eye(size)
  return K_duu

#d/du d/uz K(u,z)
def K_duz(X,Y,params,reg=False,nugget=10**-5):
  size=len(X[:,0])
  size2=len(Y[:,0])
  X0=jnp.transpose(jnp.tile(X[:,0],(size2,1)))
  Y0=jnp.transpose(jnp.tile(Y[:,0],(size,1)))

  K_duz = vmap(lambda s, t: kernel_uz(s,t,params))(X0.flatten(),np.transpose(Y0).flatten())
  K_duz = np.reshape(K_duz,(size,size2))

  if reg==True and size==size2:
    K_duz+=+nugget*jnp.eye(size)
  return K_duz

#d/dz d/du^2 K(u,z)
def K_duuz(X,Y,params,reg=False,nugget=10**-5):
  size=len(X[:,0])
  size2=len(Y[:,0])
  X0=jnp.transpose(jnp.tile(X[:,0],(size2,1)))
  Y0=jnp.transpose(jnp.tile(Y[:,0],(size,1)))

  K_duuz = vmap(lambda s, t: kernel_uuz(s,t,params))(X0.flatten(),np.transpose(Y0).flatten())
  K_duuz = np.reshape(K_duuz,(size,size2))

  if reg==True and size==size2:
    K_duuz+=+nugget*jnp.eye(size)
  return K_duuz

#d/dz^2 d/du K(u,z)
def K_duzz(X,Y,params,reg=False,nugget=10**-5):
  size=len(X[:,0])
  size2=len(Y[:,0])
  X0=jnp.transpose(jnp.tile(X[:,0],(size2,1)))
  Y0=jnp.transpose(jnp.tile(Y[:,0],(size,1)))

  K_duzz = vmap(lambda s, t: kernel_uzz(s,t,params))(X0.flatten(),np.transpose(Y0).flatten())
  K_duzz = np.reshape(K_duzz,(size,size2))

  if reg==True and size==size2:
    K_duzz+=+nugget*jnp.eye(size)
  return K_duzz

#d/dz^2 d/du^2 K(u,z)
def K_duuzz(X,Y,params,reg=False,nugget=10**-5):
  size=len(X[:,0])
  size2=len(Y[:,0])
  X0=jnp.transpose(jnp.tile(X[:,0],(size2,1)))
  Y0=jnp.transpose(jnp.tile(Y[:,0],(size,1)))

  K_duuzz = vmap(lambda s, t: kernel_uuzz(s,t,params))(X0.flatten(),np.transpose(Y0).flatten())
  K_duuzz = np.reshape(K_duuzz,(size,size2))

  if reg==True and size==size2:
    K_duuzz+=+nugget*jnp.eye(size)
  return K_duuzz

def matrix_assembly(X,params):

  #Constructs K(\phi,\phi)
  size = len(X[:,0])
  K = jnp.zeros((size+2,size+2))

  K_u = K_du(X,X,params)
  K_uu = K_duu(X,X,params)
  K_uz = K_duz(X,X,params)
  K_uzz = K_duzz(X,X,params)
  K_uuz = K_duuz(X,X,params)
  K_uuzz = K_duuzz(X,X,params)


  K = K.at[1:size+1,1:size+1].set(nu**2*K_uuzz+nu/2*K_uuz+nu/2*K_uzz+1/4*K_uz)
  K = K.at[0,0].set(kernel(0,0,params))
  K = K.at[size+1,size+1].set(kernel(1,1,params))
  K = K.at[0,size+1].set(kernel(0,1,params))
  K = K.at[size+1,0].set(K[0,size+1])

  K = K.at[1:size+1,0].set(nu*kernel_uu(X,jnp.zeros((size,1)),params)+1/2*kernel_u(X,jnp.zeros((size,1)),params))
  K = K.at[0,1:size+1].set(K[1:size+1,0])
  K = K.at[1:size+1,size+1].set(nu*kernel_uu(X,jnp.ones((size,1)),params)+1/2*kernel_u(X,jnp.ones((size,1)),params))
  K = K.at[size+1,1:size+1].set(K[1:size+1,size+1])

  return K

#Computes vector K(u,\phi)
def K_vector(X_test,X_train,params):
  size=len(X_test[:,0])
  size2 = len(X_train[:,0])

  K_vector = jnp.zeros((size,size2+2))

  X_test = X_test[:,0].reshape((size,1))
  X_train = X_train[:,0].reshape((size2,1))

  #K(u,\phi_i) where 2<=i<=N+1

  K_vector = K_vector.at[:,1:size2+1].set(nu*K_dzz(X_test,X_train,params)+1/2*K_dz(X_test,X_train,params))

  #K(u,\phi_1)
  K_vector = K_vector.at[:,0].set(kernel(X_test,jnp.zeros((size,1)),params))

  #K(u,\phi_{N+2})
  K_vector = K_vector.at[:,size2+1].set(kernel(X_test,jnp.ones((size,1)),params))

  return K_vector

#Kernel regression for one dimension K(u,\phi)K(\phi,\phi)^-1 Y
def kernel_regression(X_test, Y_train,params,K_matrix,X_train,nugget=10**-4):
    t_matrix = K_vector(X_test,X_train,params)
    size = len(K_matrix[:,0])
    prediction = jnp.matmul(t_matrix,jnp.linalg.inv(K_matrix+nugget*jnp.eye(size))@Y_train)
    return prediction
