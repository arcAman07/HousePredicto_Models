using Pkg

Pkg.add("Flux")

Pkg.add("MLDatasets")

using MLDatasets: BostonHousing

features = BostonHousing.features();

summary(features)
"13×506 Matrix{Float64}"

target = BostonHousing.targets();

using Flux

model = Dense(13,1)

model.weight

model.bias

target

features

using Flux: train!

Pkg.add("Printf")

using Printf

opt = Descent()

x_train = features[:,1:355]

x_test = features[:,356:506]

y_train = target[:,1:355]

y_test = target[:,356:506]

loss(x, y) = Flux.Losses.mse(model(x), y)

loss(x_train,y_train)

data = [(x_train,y_train)]

parameters = Flux.params(model)

# train!(loss, parameters, data, opt)

parameters

train!(loss, parameters, data, opt)

loss(x_train, y_train)

parameters

loss(x_test,y_test)

using BSON: @save

@save "mymodel.bson" model
