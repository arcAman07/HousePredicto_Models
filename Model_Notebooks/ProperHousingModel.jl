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

x_train = features[:,1:375]

x_test = features[:,376:506]

y_train = target[:,1:375]

y_test = target[:,376:506]

loss(x, y) = Flux.Losses.mse(model(x), y)

loss(x_train,y_train)

data = [(x_train,y_train)]

parameters = Flux.params(model)

# train!(loss, parameters, data, opt)

parameters

train!(loss, parameters, data, opt)

loss(x_train, y_train)

parameters

correct_pred = 0
total_pred = 0
net_loss = 0

for i in 376:506
    x = features[:,i]
    y = target[i]
    prediction = model(x)
    loss_value = loss(x,y)
    if x==y
        correct_pred += 1
    end
    total_pred += 1
    net_loss += loss_value
    println(prediction)
end
        

print(correct_pred/total_pred)

print(net_loss)

using BSON: @save

@save "mymodel.bson" model


