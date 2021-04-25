const renderInfoBox = (carInfo) => {
    let design = carInfo.design;
    
    return `<h3>${carInfo.key}</h3>
    </br>
    <table>
    <tr>
        <td>Wheel Radius: ${design.wheel_rad.toFixed(2)}</td>
        <td>Brake Sensitivity: ${design.brake_scalar.toFixed(2)}</td>
        <td>Color: ${design.color}</td>
        
    </tr>
    <tr>
        <td>Wheel Width: ${design.wheel_width.toFixed(2)}</td>
        <td>Steering Sensitivity: ${design.steering_scalar.toFixed(2)}</td>
        <td>Estimated Cost: \$${calculateCarCost(design).toFixed(2)} </td>
    </tr>
    <tr>
        <td>Tire Tread: ${design.friction_lim.toFixed(2)}</td>
        <td>Rear Steering Power: ${design.rear_steering_scalar.toFixed(2)}</td>
        <td>Car Weight: ${calculateBodyWeight(design).toFixed(2)}</td>
    </tr>
    <tr>
        <td>Engine Power: ${design.eng_power.toFixed(2)}</td>
        <td>Top Speed Limiter: ${design.max_speed.toFixed(2)}</td>
    </tr>
    </table> <div class='carsvglarge'>${renderCar(design)}<div>`
}


const transform = (coords,xOffset,yOffset,scale=1.0,rotation=90)=>{
    //for now, just rotate 90 degrees, but should make this
    //return
    //coords.map(v=>[parseInt(scale*(Math.cos(theta)*v[0]-Math.sin(theta)*v[1]))+xOffset,
    //              parseInt(scale*(Math.sin(theta)*v[0]+Math.cos(theta)*v[1]))+yOffset])
    // console.log('GOT COORDS',coords);
    return coords.map(v=>[parseInt(scale*(v[0])+xOffset), parseInt(scale*(v[1])+yOffset)])
  }

const invtransform = (coords,xOffset,yOffset,scale=2.0,rotation=90)=>{
    return coords.map(v=>[parseInt((v[0]-xOffset)*(1.0/scale)),parseInt((v[1]-yOffset)*(1.0/scale))])
}

const coords2SVG = (coords,fill,density=2.0,xOffset=275,yOffset=200,scale=2.0,className='') =>{
    //console.log("DENSITy",density)
    let offsetCoords = transform(coords,xOffset,yOffset,scale)//coords.map(x=>[parseInt(scale*(x[1])+xOffset), parseInt(scale*(x[0])+yOffset)]);
    let sortedCoords = clockwiseSort(offsetCoords);//[offsetCoords[0],offsetCoords[1], offsetCoords[3], offsetCoords[2]];//.sort((a,b)=>a[0]-b[0])
    //let pathStr = "M 10 10 ";
    //console.log("BEFORE",offsetCoords);
    //let temp = sortedCoords[1];
    //sortedCoords[1] = sortedCoords[2];
    //sortedCoords[2] = temp;
    //console.log("AFTER",sortedCoords);
    let pathStr = "M "+sortedCoords[0][0]+" "+sortedCoords[0][1];
    for (let c=1; c<sortedCoords.length; c++){
      pathStr += " L "+sortedCoords[c][0]+" "+sortedCoords[c][1];
    }
    //close the path
    pathStr += " L "+sortedCoords[0][0]+" "+sortedCoords[0][1];
    //console.log(pathStr);
    if (className != ''){
        return `<path d="${pathStr}" fill="#${fill}" class=${className} stroke="black" fillOpacity=${parseFloat(density)/1.0} />`
    }
        
    else{
        return `<path d="${pathStr}" fill="#${fill}" stroke="black" fillOpacity=${parseFloat(density)/1.0} />`
    }
}

const  wheelattrs2Coords = (wheel_r, wheel_w, wheel_pos) => {
    return [[wheel_pos[0]-wheel_w/2, wheel_pos[1]+wheel_r],[wheel_pos[0]+wheel_w/2, wheel_pos[1]+wheel_r],
      [wheel_pos[0]-wheel_w/2, wheel_pos[1]-wheel_r],[wheel_pos[0]+wheel_w/2, wheel_pos[1]-wheel_r]]
}

//some math functions b/c js doesn't support basic stuff like adding arrays
const arrSum = (a1,a2) => {
    return a1.map((val,i)=>(val+a2[i]));
  }
  
const arrDiff = (a1,a2) => {
    return a1.map((val,i)=>(val-a2[i]));
  }
const arrListSum = (arrList) =>{
    return arrList.reduce((a,c)=>arrSum(a,c),[0,0]);
  }
  
const arrListMean = (arrList) =>{
    return arrListSum(arrList).map(a=>a/(1.0*arrList.length))
  }
  
  //sorts the vertices of a CONVEX polygon in clockwise order
const clockwiseSort = (vertices) => {
    let interiorPoint = arrListMean(vertices);
    // console.log(interiorPoint);
    let vectors = vertices.map(v=>arrDiff(v,interiorPoint));
    // console.log(vectors);
    let vertsWithRad = vectors.map((v,i)=>[...vertices[i],Math.atan2(v[1],v[0])])
    // console.log(vertsWithRad);
    return vertsWithRad.sort((a,b)=>b[2]-a[2]).map(v=>v.slice(0,2));
  }
  

const renderCar = (carConfig) => {
    let scale = 2.0;
    let wheelColor = "000000";
    let carWidth = 500;
    let carLength = 500;
    let bumper = carConfig.hull_poly1;
    let hull1 = carConfig.hull_poly2;
    let hull2 = carConfig.hull_poly3;
    let spoiler = carConfig.hull_poly4;
    let wheels = carConfig.wheel_pos;
    let wheel_coords = wheels.map(w => wheelattrs2Coords(carConfig.wheel_rad, carConfig.wheel_width, w)); 
    let xOffset = carWidth/2;
    let yOffset = carLength/2;
    let bumperCoords = clockwiseSort(transform(bumper,0,yOffset+30,scale));
    let hull1Coords = clockwiseSort(transform(hull1,xOffset+175,0,scale));
    let className = 'carBody';
    let SVGResult;

      let wheelPairs = [];
      if(wheels){
        let maxWheelPosY = wheels.reduce((a,w)=>Math.max(a,w[1]),0);
        let minWheelPosY = wheels.reduce((a,w)=>Math.min(a,w[1]),maxWheelPosY);
        let frontPair = wheels.filter(w=>w[1]==minWheelPosY); //not sure how to initialize min, just did a big number
        let rearPair = wheels.filter(w=>w[1]==maxWheelPosY);
        // console.log("FRONT PAIR",frontPair);
        // console.log("REAR PAIR", rearPair);
        wheelPairs = [frontPair,rearPair].map((pair)=>transform(pair,xOffset,yOffset,scale));
      }

    //   SVGResult = "<svg><line x1=10 y1=10 x2=50 y2=50 strokeWidth=5 stroke='black'/></svg>"
    SVGResult = `<svg viewbox="0 0 600 600"><line x1=${wheelPairs[0][0][0]+carConfig.wheel_width/2} y1=${wheelPairs[0][0][1]} ` +
                    `x2=${wheelPairs[0][1][0]-carConfig.wheel_width/2} y2=${wheelPairs[0][1][1]} ` +
                    `strokeWidth=${5} stroke="${wheelColor}"/>` +
                `<line x1=${wheelPairs[1][0][0]+carConfig.wheel_width/2} y1=${wheelPairs[1][0][1]} ` +
                    `x2=${wheelPairs[1][1][0]-carConfig.wheel_width/2} y2=${wheelPairs[1][1][1]} ` +
                    `strokeWidth=${5} stroke="${wheelColor}"/>`+
                `${coords2SVG(bumper,carConfig.color,carConfig.hull_densities[0],xOffset,yOffset,scale,className)}` + 
                `${coords2SVG(hull1,carConfig.color,carConfig.hull_densities[1],xOffset,yOffset,scale,className)}` + 
                `${coords2SVG(hull2,carConfig.color,carConfig.hull_densities[2],xOffset,yOffset,scale,className)}` +
                `${coords2SVG(spoiler,carConfig.color,carConfig.hull_densities[3],xOffset,yOffset,scale,className)}` +
                `${wheel_coords.map(w=>coords2SVG(w,wheelColor,0.6+(0.25*carConfig.friction_lim/1e4),xOffset,yOffset))}</svg>`
    // console.log(SVGResult);
    return SVGResult;
}


// let bumper = this.props.config ? this.props.config.hull_poly1 : undefined;
// let hull1 = this.props.config ? this.props.config.hull_poly2 : undefined;
// let hull2 = this.props.config ? this.props.config.hull_poly3 : undefined;
// let spoiler = this.props.config ? this.props.config.hull_poly4 : undefined;
// let wheels = this.props.config ? this.props.config.wheel_pos : undefined;
// let wheel_coords = wheels ? wheels.map(w => this.wheelattrs2Coords(this.props.config.wheel_rad, this.props.config.wheel_width, w)) : [];
// let xOffset = this.props.width ? this.props.width/2 : undefined;
// let yOffset = this.props.height ? this.props.height/2 : undefined;
// let carLength = this.props.carLength;
// let carWidth = this.props.carWidth;
// let bumperCoords = bumper ? clockwiseSort(transform(bumper,0,yOffset+30,2.0)) : undefined;
// let hull1Coords = hull1 ? clockwiseSort(transform(hull1,xOffset+175,0,2.0)) : undefined;
// let SVGResult;

//   let wheelPairs = [];
//   if(wheels){
//     let maxWheelPosY = wheels.reduce((a,w)=>Math.max(a,w[1]),0);
//     let minWheelPosY = wheels.reduce((a,w)=>Math.min(a,w[1]),maxWheelPosY);
//     let frontPair = wheels.filter(w=>w[1]==minWheelPosY); //not sure how to initialize min, just did a big number
//     let rearPair = wheels.filter(w=>w[1]==maxWheelPosY);
//     // console.log("FRONT PAIR",frontPair);
//     // console.log("REAR PAIR", rearPair);
//     wheelPairs = [frontPair,rearPair].map((pair)=>transform(pair,xOffset,yOffset,2.0));
//   }


//   {wheels && <line x1={wheelPairs[0][0][0]+this.props.config.wheel_width/2} y1={wheelPairs[0][0][1]}
//   x2={wheelPairs[0][1][0]-this.props.config.wheel_width/2} y2={wheelPairs[0][1][1]}
//   strokeWidth={5} stroke={this.props.wheelColor}/>}
// {wheels && <line x1={wheelPairs[1][0][0]+this.props.config.wheel_width/2} y1={wheelPairs[1][0][1]}
//   x2={wheelPairs[1][1][0]-this.props.config.wheel_width/2} y2={wheelPairs[1][1][1]}
//   strokeWidth={5} stroke={this.props.wheelColor}/>}
// {bumper && this.coords2SVG(bumper,this.props.hullColor,this.props.config.hull_densities[0],xOffset,yOffset)}
// {hull1 && this.coords2SVG(hull1,this.props.hullColor,this.props.config.hull_densities[1],xOffset,yOffset)}
// {hull2 && this.coords2SVG(hull2,this.props.hullColor,this.props.config.hull_densities[2],xOffset,yOffset)}
// {spoiler && this.coords2SVG(spoiler,this.props.hullColor,this.props.config.hull_densities[3],xOffset,yOffset)}
// {wheel_coords.map(w=>this.coords2SVG(w,this.props.wheelColor,0.6+(0.25*this.props.config.friction_lim/1e4),xOffset,yOffset))}