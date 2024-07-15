using Pkg
Pkg.activate(".")

using Agents, Agents.Pathfinding
using Random
import ImageMagick
using FileIO: load
using GLMakie

@multiagent :opt_speed struct Animal(ContinuousAgent{3,Float64})
    @subagent struct Rabbit
        energy::Float64
    end
    @subagent struct Fox
        energy::Float64
    end
    @subagent struct Hawk
        energy::Float64
    end
end

eunorm(vec) = √sum(vec .^ 2)
const v0 = (0.0, 0.0, 0.0) 


function initialize_model(
    heightmap_url =
    "https://raw.githubusercontent.com/JuliaDynamics/" *
    "JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png",
    water_level = 8,
    grass_level = 20,
    mountain_level = 35;
    n_foxes = 30, 
    n_rabbits = 160,
    n_hawks = 30, 
    Δe_grass = 25, 
    Δe_rabbit = 30,  
    rabbit_repr = 0.06, 
    fox_repr = 0.03,  
    hawk_repr = 0.02, 
    rabbit_vision = 6,  
    fox_vision = 10, 
    hawk_vision = 15, 
    rabbit_speed = 1.3,
    fox_speed = 1.1, 
    hawk_speed = 1.2,
    regrowth_chance = 0.03, 
    dt = 0.1,  
    seed = 42,  
)

    heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1
    
    dims = (size(heightmap)..., 50)
    
    land_walkmap = BitArray(falses(dims...))
    air_walkmap = BitArray(falses(dims...))
    for i in 1:dims[1], j in 1:dims[2]

        if water_level < heightmap[i, j] < grass_level
            land_walkmap[i, j, heightmap[i, j]+1] = true
        end
        
        if heightmap[i, j] < mountain_level
            air_walkmap[i, j, (heightmap[i, j]+1):mountain_level] .= true
        end
    end

    rng = MersenneTwister(seed)

    space = ContinuousSpace((100., 100., 50.); periodic = false)

    grass = BitArray(
        rand(rng, dims[1:2]...) .< ((grass_level .- heightmap) ./ (grass_level - water_level)),
    )
    properties = (
        landfinder = AStar(space; walkmap = land_walkmap),
        airfinder = AStar(space; walkmap = air_walkmap, cost_metric = MaxDistance{3}()),
        Δe_grass = Δe_grass,
        Δe_rabbit = Δe_rabbit,
        rabbit_repr = rabbit_repr,
        fox_repr = fox_repr,
        hawk_repr = hawk_repr,
        rabbit_vision = rabbit_vision,
        fox_vision = fox_vision,
        hawk_vision = hawk_vision,
        rabbit_speed = rabbit_speed,
        fox_speed = fox_speed,
        hawk_speed = hawk_speed,
        heightmap = heightmap,
        grass = grass,
        regrowth_chance = regrowth_chance,
        water_level = water_level,
        grass_level = grass_level,
        dt = dt,
    )

    model = StandardABM(Animal, space; agent_step! = animal_step!, 
                        model_step! = model_step!, rng, properties)

    for _ in 1:n_rabbits
        pos = random_walkable(model, model.landfinder)
        add_agent!(pos, Rabbit, model, v0, rand(abmrng(model), Δe_grass:2Δe_grass))
    end
    for _ in 1:n_foxes
        pos = random_walkable(model, model.landfinder)
        add_agent!(pos, Fox, model, v0, rand(abmrng(model), Δe_rabbit:2Δe_rabbit))
    end
    for _ in 1:n_hawks
        pos = random_walkable(model, model.airfinder)
        add_agent!(pos, Hawk, model, v0, rand(abmrng(model), Δe_rabbit:2Δe_rabbit))
    end
    return model
end


@dispatch function animal_step!(rabbit::Rabbit, model)
    if get_spatial_property(rabbit.pos, model.grass, model) == 1
        model.grass[get_spatial_index(rabbit.pos, model.grass, model)] = 0
        rabbit.energy += model.Δe_grass
    end

    rabbit.energy -= model.dt
    
    if rabbit.energy <= 0
        remove_agent!(rabbit, model, model.landfinder)
        return
    end


    predators = [
        x.pos for x in nearby_agents(rabbit, model, model.rabbit_vision) if
            kindof(x) == :fox || kindof(x) == :hawk
            ]

    if !isempty(predators) && is_stationary(rabbit, model.landfinder)
        direction = (0., 0., 0.)
        for predator in predators
            away_direction = (rabbit.pos .- predator)

            all(away_direction .≈ 0.) && continue

            direction = direction .+ away_direction ./ eunorm(away_direction) ^ 2
        end
        if all(direction .≈ 0.)
            chosen_position = random_walkable(rabbit.pos, model, model.landfinder, model.rabbit_vision)
        else
            direction = direction ./ eunorm(direction)
            position = rabbit.pos .+ direction .* (model.rabbit_vision / 2.)
            chosen_position = random_walkable(position, model, model.landfinder, model.rabbit_vision / 2.)
        end
        plan_route!(rabbit, chosen_position, model.landfinder)
    end


    rand(abmrng(model)) <= model.rabbit_repr * model.dt && reproduce!(rabbit, model)

    if is_stationary(rabbit, model.landfinder)
        plan_route!(
            rabbit,
            random_walkable(rabbit.pos, model, model.landfinder, model.rabbit_vision),
            model.landfinder
        )
    end

    move_along_route!(rabbit, model, model.landfinder, model.rabbit_speed, model.dt)
end

@dispatch function animal_step!(fox::Fox, model)
    food = [x for x in nearby_agents(fox, model) if kindof(x) == :rabbit]
    if !isempty(food)
        remove_agent!(rand(abmrng(model), food), model, model.landfinder)
        fox.energy += model.Δe_rabbit
    end

    fox.energy -= model.dt
    
    if fox.energy <= 0
        remove_agent!(fox, model, model.landfinder)
        return
    end

    rand(abmrng(model)) <= model.fox_repr * model.dt && reproduce!(fox, model)

    if is_stationary(fox, model.landfinder)
        prey = [x for x in nearby_agents(fox, model, model.fox_vision) if kindof(x) == :rabbit]
        if isempty(prey)
            plan_route!(
                fox,
                random_walkable(fox.pos, model, model.landfinder, model.fox_vision),
                model.landfinder,
            )
        else
            plan_route!(fox, rand(abmrng(model), map(x -> x.pos, prey)), model.landfinder)
        end
    end
    
    move_along_route!(fox, model, model.landfinder, model.fox_speed, model.dt)
end

@dispatch function animal_step!(hawk::Hawk, model)
    food = [x for x in nearby_agents(hawk, model) if kindof(x) == :rabbit]
    if !isempty(food)
        remove_agent!(rand(abmrng(model), food), model, model.airfinder)
        hawk.energy += model.Δe_rabbit
        plan_route!(hawk, hawk.pos .+ (0., 0., 7.), model.airfinder)
    end

    hawk.energy -= model.dt
    if hawk.energy <= 0
        remove_agent!(hawk, model, model.airfinder)
        return
    end

    rand(abmrng(model)) <= model.hawk_repr * model.dt && reproduce!(hawk, model)

    if is_stationary(hawk, model.airfinder)
        prey = [x for x in nearby_agents(hawk, model, model.hawk_vision) if kindof(x) == :rabbit]
        if isempty(prey)
            plan_route!(
                hawk,
                random_walkable(hawk.pos, model, model.airfinder, model.hawk_vision),
                model.airfinder,
            )
        else
            plan_route!(hawk, rand(abmrng(model), map(x -> x.pos, prey)), model.airfinder)
        end
    end

    move_along_route!(hawk, model, model.airfinder, model.hawk_speed, model.dt)
end


function reproduce!(animal, model)
    animal.energy = Float64(ceil(Int, animal.energy / 2))
    add_agent!(animal.pos, eval(kindof(animal)), model, v0, animal.energy)
end


function model_step!(model)
    growable = view(
        model.grass,
        model.grass .== 0 .& model.water_level .< model.heightmap .<= model.grass_level,
    )
    growable .= rand(abmrng(model), length(growable)) .< model.regrowth_chance * model.dt
end

model = initialize_model()


@dispatch animalcolor(a::Rabbit) = :brown
@dispatch animalcolor(a::Fox) = :orange
@dispatch animalcolor(a::Hawk) = :blue


const ABMPlot = Agents.get_ABMPlot_type()
function Agents.static_preplot!(ax::Axis3, p::ABMPlot)
    surface!(
        ax,
        (100/205):(100/205):100,
        (100/205):(100/205):100,
        p.abmobs[].model[].heightmap;
        colormap = :terrain
    )
end

fig, ax, abmobs = abmplot(model;
    agent_color = animalcolor, 
    agent_size = 1.0,
    add_controls = true,)


fig

# abmvideo(
#     "rabbit_fox_hawk_model.mp4",
#     model;
#     figure = (size = (800, 700),),
#     frames = 300,
#     framerate = 15,
#     agent_color = animalcolor,
#     agent_size = 1.0,
#     title = "Rabbit Fox Hawk"
# )
