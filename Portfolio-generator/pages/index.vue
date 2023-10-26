<template>

    <div class="HomeContent">
        <div class="HomeContent__container">
                <h1 class="AboutMeTitle">Hi, I'm {{FormData?.nick}}</h1>
                <h2 class="AboutMeContent">{{ FormData?.about_me }}</h2>
            </div>

    </div>

</template>

<script lang="ts">

export default {
    name: 'Home',
    props: {
        FormData: Object,
    },
}

</script>

<style scoped>

.HomeContent {
    display: flex;
    flex-direction: column;
    height: 100%;
}

.HomeContent__container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
}

.AboutMeTitle {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-align: center;
    vertical-align: top;
    
}

.AboutMeContent {
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 1rem;
    text-align: center;
}
</style>