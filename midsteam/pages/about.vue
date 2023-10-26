<template>
    <div class="About">
        <h1>About us</h1>
        <p>Our company is a small indie game studio based in the UK. We are a team of 3 people who are passionate about making games. We are currently working on our first game, which is a 2D platformer. We are hoping to release it in 2021. As a means to get by we developed a middleman website to bridge the gap between the store and the customer.</p>

        <h2>Our team</h2>

        <div class="About__team">
            <div class="About__team__member">

                <h3>Felix Manning</h3>
                <p>Felix is the founder of the company. They are a programmer and a game designer. They are responsible for the game mechanics and the overall design of the game.</p>
            </div>
            <div class="About__team__member">

                <h3>Jane Doe</h3>
                <p>Jane is a programmer and a game designer. She is responsible for the game mechanics and the overall design of the game.</p>
            </div>
            <div class="About__team__member">

                <h3>Jack Doe</h3>
                <p>Jack is a programmer and a game designer. He is responsible for the game audiovisual side and the overall feel of the game.</p>
            </div>

        </div>


    </div>
</template>

<style scoped>

.About {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
    height: 100%;
    overflow: scroll;
}

.About__team {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

.About__team__member {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 200px;
    margin: 20px;
}

.About__team__member img {
    width: 100%;
    border-radius: 50%;
}


</style>