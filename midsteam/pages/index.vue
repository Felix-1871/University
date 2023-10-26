<template>
    <div class="HomeContent">
        <h1 style="justify-content: center; align-items: center; display: flex;">Home</h1>
        <div class="heroCatchphrase">
            <h2>Find the best deals on games.</h2>
           <h3>"Level up your game shopping experience with our Navgar magic! Get the hottest titles at the coolest prices, and let the thrill of gaming be just a click away!"
</h3>
        </div>

      <NuxtLink to="/games">  <button class="Clickme">Click here to enter a world better than you've ever imagined. The world of witchcraft, magic, and rituals</button></NuxtLink>
        <div class="imgs">
            <img src="/cod.jpg" alt="cod">
            <img src="/ink.jpg" alt="ink">
            <img src="/uno.jpg" alt="uno">
            <img src="/whi.jpg" alt="whi">
            <img src="/gom.jpg" alt="gom">

        </div>
    </div>
</template>

<style scoped>

.HomeContent {
    align-items: center;
    justify-content: center;
    height: 100vh;
    border: 2px solid #fff;
    border-radius: 10px;
    background-color: var(--ctp-mocha-crust);
}

.heroCatchphrase {
    align-items: center;
    justify-content: center;
    height: 300px;
    border-top: 2px solid #fff;
    border-radius: 10px;
    background-color: var(--ctp-mocha-crust);
}

.Clickme {
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 200px;
    background-color: #595988;
    color: #fff;
    font-weight: bold;
    border: 2px solid #fff;
    box-shadow: inset 0 0 10px #000;
    border-radius: 90px;

}

.Clickme:hover {
    background-color: #fff;
    color: #595988;
    border: 2px solid #595988;
    box-shadow: inset 0 0 10px #000;
    border-radius: 90px;
}

.imgs {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 200px;

    border-radius: 90px;
    margin-top: 20px;
}

.imgs img {
    width: 200px;
    height: 200px;
    margin: 10px;
    border-radius: 10px;
}






</style>