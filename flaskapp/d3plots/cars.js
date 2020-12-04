let w = 1250,
  h = 800,
  pad = 50,
  left_pad = 100,
  data_url = "../car_list.json",
  annotation_offset = 5;
  xmeasure = meanRewards;

//the trajectory window over which we are calculating
let xwindow = ywindow = 100;
  
let xfunc = xmeasure.f,
  xlabel = xmeasure.label(xwindow),
  xrange = xmeasure.range,
  yfunc = ymeasure.f,
  ylabel = ymeasure.label(ywindow),
  yrange = ymeasure.range;

let human_color = 'blue';
let hybrid_color = (intensity)=> ('#' + ('000000' + (intensity* 0x100).toString(16)).slice(-6));
let bo_color = 'red';
let default_color = 'black';

let svg = d3
    .select("#scatter")
    .append("svg")
    .attr("width", w)
    .attr("height", h);


let x = d3.scaleLinear()
    .domain(xrange)
    .range([left_pad, w - pad]);

let y = d3.scaleLinear()
    .domain(yrange)
    .range([h - pad, pad]);

let xAxis = d3.axisBottom().scale(x),
    yAxis = d3.axisLeft().scale(y);



// multiplot

let multi_pad_left = 50;
let multi_pad_bottom = 50;
let multi_pad_top = 20;

let multi_x = d3.scaleLinear()
    .domain([0,500])
    .range([0,300]);

let multi_y = d3.scaleLinear()
    .domain([0,1000])
    .range([150,0]);


let multixAxis = d3.axisBottom().scale(multi_x);
let multiyAxis = d3.axisLeft().scale(multi_y);

svg
    .append("g")
    .attr("class", "axis")
    .attr("transform", "translate(0, " + (h - pad) + ")")
    .call(xAxis);

svg
    .append("text")
    .attr("transform", "translate(" + (w-725)  + ", " + (h-5) + ")")
    .text(xlabel);

svg
    .append("text")
    .attr("transform", `rotate(-90) translate(${-500},${40})`)
    .text(ylabel);

svg
    .append("g")
    .attr("class", "axis")
    .attr("transform", "translate(" + (left_pad) + ", 0)")
    .call(yAxis);

svg
    .append("text")
    .attr("class", "loading")
    .text("Loading ...")
    .attr("x", function () {
      return w / 2;
    })
    .attr("y", function () {
      return h / 2 - 5;
    });

let x_cursor_group = svg.append("g");
let x_cursor = x_cursor_group.append("path")
    .attr("d", d3.line()([[left_pad+50,0+pad],[left_pad+50,h-pad]]))
    .attr("stroke","black");
let x_cursor_text = x_cursor_group.append("text")
    .attr("y",h-20);

let y_cursor_group = svg.append("g");
let y_cursor = y_cursor_group.append("path")
    .attr("d", d3.line()([[left_pad,200+pad],[w-pad,200+pad]]))
    .attr("stroke","black");
let y_cursor_text = y_cursor_group.append("text")
    .attr("x",left_pad-50);

let hover_text = svg.append("text")
    .attr("x",left_pad+55)
    .attr("y",195+pad)
    .text("Hover on a car for details.")

let infobox = d3.select("body").append("div")
    .attr("class","tooltip")
    .style("opacity",0)


let selected_groups = ['bo','human','hybrid_hum_select', 'hybrid_hum_init', 
                        'hybrid_hum_init_select', 'hybrid_hum_confidence', 'default'];


let legend = svg.append("g")
    .attr("class","legend")


let update_selected_groups = (group) => {
    if(selected_groups.includes(group)){
        console.log("deleting")
        selected_groups.splice(selected_groups.indexOf(group),1);
    }
    else{
        selected_groups.push(group);
    }   
    let c = legend.selectAll("circle")
        .each(function(d,i){
            let circ = d3.select(this)
            console.log(circ);
            let opacity = selected_groups.includes(circ.attr("type")) ? 1 : 0.35;
            circ.attr("opacity",opacity);
        // .attr("opacity",(d)=>{
        //     console.log("D",d);
        //     if(selected_groups.includes(circ.attr("type"))){
        //         return 1;
        //     }
        //     else{
        //         return 0.35;
        //     }
        // })
    });
    d3.selectAll("g.data")
        .attr("visibility",(d)=>{
            if(selected_groups.includes(d.type))
                return "visible";
            else
                return "hidden";
        });
    console.log(selected_groups);
}
legend
    .append("rect")
    .attr("x", w-250-legend_x_margin)
    .attr("y", legend_y_margin + 50)
    .attr("width", 300)
    .attr("height", 300)
    .attr("fill","white")
    .attr("stroke","black")
    // .on("click",function(e){
    //     update_selected_groups("bo");
    //     update_selected_groups("human");
    //     update_selected_groups("hybrid");
    //     update_selected_groups("default");
    // })
legend
    .append("circle")
    .attr("cx", w-220-legend_x_margin)
    .attr("cy", legend_y_margin + 70)
    .attr("r", 10)
    .attr("fill",bo_color)
    .attr("stroke","black")
    .attr("opacity",1)
    .attr("type","bo")
    .on("click",function(e){
        let c = d3.select(e.target);
        // if(c.attr("opacity") == 1){
        //     //deselect
        //     c.attr("opacity",0.35);
        //     d3.selectAll("g.data")
        //     .attr("visibility","true");
        // }
        // else{
        //     c.attr("opacity",1);
        update_selected_groups("bo");
            // d3.selectAll("g.data")
            // .attr("visibility",(d)=>{
            //     if(d.type=="bo")
            //         return "visible";
            //     else
            //         return "hidden";
            // });
        // }
    });
legend
    .append("text")
    .attr("x", w-200-legend_x_margin)
    .attr("y", legend_y_margin + 75)
    .text("AI Designs");

legend
    .append("circle")
    .attr("cx", w-220-legend_x_margin)
    .attr("cy", legend_y_margin + 105)
    .attr("r", 10)
    .attr("fill",human_color)
    .attr("stroke","black")
    .attr("opacity",1)
    .attr("type","human")
    .on("click",function(e){
        update_selected_groups("human");
    });
legend
    .append("text")
    .attr("x", w-200-legend_x_margin)
    .attr("y", legend_y_margin + 110)
    .text("Human Designs");
///////////////////////////////////////////////////
legend
    .append("circle")
    .attr("cx", w-220-legend_x_margin)
    .attr("cy", legend_y_margin + 140)
    .attr("r", 10)
    .attr("fill",hybrid_color(255))
    .attr("stroke","black")
    .attr("opacity",1)
    .attr("type","hybrid_hum_select")
    .on("click",function(e){
        update_selected_groups("hybrid_hum_select");
    });
legend
    .append("text")
    .attr("x", w-200-legend_x_margin)
    .attr("y", legend_y_margin + 145)
    .text("Hybrid: Human-Selected Features");
///////////////////////////////////////////////////

///////////////////////////////////////////////////
legend
    .append("circle")
    .attr("cx", w-220-legend_x_margin)
    .attr("cy", legend_y_margin + 175)
    .attr("r", 10)
    .attr("fill",hybrid_color(205))
    .attr("stroke","black")
    .attr("opacity",1)
    .attr("type","hybrid_hum_init")
    .on("click",function(e){
        update_selected_groups("hybrid_hum_init");
    });
legend
    .append("text")
    .attr("x", w-200-legend_x_margin)
    .attr("y", legend_y_margin + 180)
    .text("Hybrid: Human-Initialized Design");
///////////////////////////////////////////////////

///////////////////////////////////////////////////
legend
    .append("circle")
    .attr("cx", w-220-legend_x_margin)
    .attr("cy", legend_y_margin + 210)
    .attr("r", 10)
    .attr("fill",hybrid_color(155))
    .attr("stroke","black")
    .attr("opacity",1)
    .attr("type","hybrid_hum_init_select")
    .on("click",function(e){
        update_selected_groups("hybrid_hum_init_select");
    });
legend
    .append("text")
    .attr("x", w-200-legend_x_margin)
    .attr("y", legend_y_margin + 215)
    .text("Hybrid: Human Features and Design");
///////////////////////////////////////////////////

///////////////////////////////////////////////////
legend
    .append("circle")
    .attr("cx", w-220-legend_x_margin)
    .attr("cy", legend_y_margin + 245)
    .attr("r", 10)
    .attr("fill",hybrid_color(75))
    .attr("stroke","black")
    .attr("opacity",1)
    .attr("type","hybrid_hum_confidence")
    .on("click",function(e){
        update_selected_groups("hybrid_hum_confidence");
    });
legend
    .append("text")
    .attr("x", w-200-legend_x_margin)
    .attr("y", legend_y_margin + 250)
    .text("Hybrid: Confident Features");
///////////////////////////////////////////////////



legend
    .append("circle")
    .attr("cx", w-220-legend_x_margin)
    .attr("cy", legend_y_margin + 280)
    .attr("r", 10)
    .attr("fill",default_color)
    .attr("stroke","black")
    .attr("opacity",1)
    .attr("type","default")
    .on("click",function(e){
        update_selected_groups("default");
    });
legend
    .append("text")
    .attr("x", w-200-legend_x_margin)
    .attr("y", legend_y_margin + 285)
    .text("Default Design");

let cars_dict = {};



d3.json(data_url)
    .then((cars) => {
    for(let i=0; i<cars.length; i++){
        cars_dict[cars[i].key] = cars[i];
    }
    svg.selectAll(".loading").remove();
    let points = svg
        .selectAll("g.data")
        .data(cars)
        .enter()
        .append("g")
        .attr("class", "data")
        .attr("id", (d)=> d.key+"_group");
        // .attr("x", (d) => {
        //     return xfunc(d['results'],x,window=100);
        // })
        // .attr("y", (d) => {
        //     return yfunc(d['results'],y,window=100);
        // })
        // .attr("height", 10)
        // .attr("width", 10)
        // .attr("transform", (d) => {
        //     console.log("OHOHOHO")
        //     let rewards = d['results']
        //     let meanRewardSum = 0;
        //     for(let i=0; i<rewards.length; i++){
        //         meanRewardSum += jStat.mean(rewards[i]);
        //     }
        //     return "translate("+meanRewardSum/rewards.length + "," + d['results'][0][1]+")";
        // });

    points
        .append("circle")
        .attr("class", "circle")
        .attr("id", (d) => d.key)
        .attr("cx", (d) => {
            return xfunc(d,x,window=100);
        })
        .attr("cy", (d) => {
            return yfunc(d,y,window=100);
        })
        .attr("r", (d) => {
            return 30;
        })
        .attr("fill", (d) => {
            if(d.type == 'human') {
                return human_color;
            }
            else if(d.type == 'default') {
                return default_color;
            }
            else if(d.type == 'bo') {
                return bo_color;
            }
            else if(d.type == 'hybrid_hum_select'){
                return hybrid_color(255);
            }
            else if(d.type == 'hybrid_hum_init'){
                return hybrid_color(205);
            }
            else if(d.type == 'hybrid_hum_init_select'){
                return hybrid_color(155);
            }
            else {
                return hybrid_color(75);
            }
        })
        .attr("opacity",0.15)
        .attr("stroke-width", 2)
        // .attr("fill", (d) =>{
        //     return "None";
        // })
        .attr('design',(d) =>{
            return d.design;
        })
        .attr('name', (d) =>{
            return d.key;
        });
        // .on("mouseover", function(e){
        //     d3.select(this).transition()
        //         .duration(200)
        //         .attr("r",50);
        // })
        // .on("mouseout", function(e){
        //     d3.select(this).transition()
        //         .duration(200)
        //         .attr("r",25);
        // });
    
    points
        .append("svg")
        .attr("class","carsvg")
        // .style("left", (d) => xfunc(d['results'],x,window=100) + "px")
        // .style("top", (d) => yfunc(d['results'],y,window=100) + "px")
        .html((d) => renderCar(d.design))
        .attr("x", (d) => {
            return xfunc(d,x,window=100)-20;
        })
        .attr("y", (d) => {
            return yfunc(d,y,window=100)-22;
        })
        .attr("height", 50)
        .attr("width", 50)
        .on("mouseover", function(e){
            hover_text.attr("fill","None")
            // d3.select(this).selectAll(".carBody")
            //     // .attr("fill", "red")
            //     .attr("stroke-width",15)
            //     .attr("stroke","gray");
            let car = e.target.parentNode.parentNode.__data__;
            let x_val = xfunc(car,x,window=100);
            let y_val = yfunc(car,y,window=100)
            x_cursor
                .attr("d", d3.line()([[x_val,0+pad],[x_val,h-pad]]))
            x_cursor_text
                .attr("x",x_val)
                .text(xmeasure.f_raw(car,window=100).toFixed(2))
            y_cursor
                .attr("d", d3.line()([[left_pad,y_val],[w-pad,y_val]]))
            y_cursor_text
                .attr("y",y_val)
                .text(ymeasure.f_raw(car,window=100).toFixed(2))
            d3.select(this).transition()
                .duration(500)
                .attr("x",x_val-20-20)
                .attr("y",y_val-20-20)
                .attr("width",100)
                .attr("height",100);
            d3.select("#"+car.key).transition()
                .duration(500)
                .attr("r",90).transition()
                .duration(200)
                .attr("r",60);
            infobox.transition()
                .duration(200)
                .style("opacity", 0.95);
            // let car = e.target.__data__;
            
            // console.log(e.target.cx.baseVal.value, e.target.cy.baseVal.value)
            infobox.html(renderInfoBox(car))
            // infobox.html(renderCar(car.design))
            .style("left", Math.min(900,e.x+10) + "px")
            .style("top", Math.max(10,e.y-400) + "px");
            console.log("HIHIHIH");
            console.log(car.results);
            //first do the running average
            let smoothedData = [];
            for(let i=0; i<car.results.length; i++){
                smoothedData.push(runningAvg(car.results[i]))
            }
            multiplot_container = infobox.append('svg')
                .attr('viewbox', '0 0 900 900')
                .attr('class','multiplot');

            
            let multiplot = multiplot_container.append('svg')
            
            multiplot.append("g")
                .attr("class", "axis")
                .attr("transform", "translate("+ multi_pad_left + ", " + (150 + multi_pad_top) + ")")
                .call(multixAxis);

            multiplot.append("g")
                .attr("class", "axis")
                .attr("transform", "translate(" + multi_pad_left + ", " + multi_pad_top + ")")
                .call(multiyAxis);

            multiplot.append("text")
                .attr("x",150)
                .attr("y",20)
                .text(`Learning Curves (n=${car.results.length})`)
            
            multiplot.append("text")
                .attr("x",140)
                .attr("y",167)
                .text("Training Episode Number")

            multiplot.append("text")
                .attr("x",-115)
                .attr("y",15)
                .attr("transform", "rotate(-90)")
                
                .text("Episode Reward")

            multiplot.selectAll(".line")
                .data(smoothedData)
                .enter()
                .append("path")
                    .attr("fill","none")
                    .attr("stroke", "lightgreen")
                    .attr("stroke-width", 1.5)
                    .attr("d", function(d){
                        // console.log(d)
                        return d3.line()
                          .x(function(d,i) { return (multi_x(i) + multi_pad_left); })
                          .y(function(d,i) { return (multi_y(d) + multi_pad_top); })
                          (d)
                      })

    
                
        })
        .on("mouseout", function(e){
            let car = e.target.parentNode.parentNode.__data__;
           
            // d3.select(this).selectAll(".carBody")
            //     // .attr("fill", `#${car.design.color}`)
            //     .attr("stroke-width",5)
            //     .attr("stroke","none");
            d3.select(this).transition()
                .duration(200)
                .attr("x",xfunc(car,x,window=100)-20)
                .attr("y",yfunc(car,y,window=100)-22)
                .attr("width",50)
                .attr("height",50);
            d3.select("#"+car.key).transition()
                .duration(200)
                .attr("r",25);
            infobox.transition()
                .duration(600)
                .style("opacity", 0);
        });


        // infobox.html(renderCar(car.design))
        // .style("left", e.target.cx.baseVal.value+10 + "px")
        // .style("top", e.target.cy.baseVal.value-250 + "px");
  
    // points
    //     .append("foreignObject")
    //     .attr("x", (d) => {
    //         return xfunc(d['results'],x,window=100)+annotation_offset;
    //     })
    //     .attr("y", (d) => {
    //         return yfunc(d['results'],y,window=100)-annotation_offset;
    //     })
    //     .attr('class','node')
    //     .attr("height",50)
    //     .attr("width",50)
    //     .attr("font-size", "10px")
    //     .append("div")
    //     .html( (d) => {
    //         return d['key'];
    //     });
        // return points;
    })
    .catch((error) =>{
        console.log(error);
    });