<template>
    <div class="ContactContent">
        <div class="Container">

                <div class="ContainerBody">
                    <div class="BodyInfo">
                        <h1>How to contact me</h1>
                        <h2>Email</h2>
                        <p> {{FormData?.contact_email}}</p>
                        <h2>Discord</h2>
                        <p> <a :href="FormData?.contact_discord" target="_blank">{{ name }}</a></p>
                    </div>
            </div>
        </div>

    </div>
</template>

<style scoped>

.ContactContent h1 {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-align: center;

}

.ContactContent h2 {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-align: center;
}

.ContactContent p {
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 1rem;
    text-align: center;
}

.ContactContent a {
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 1rem;
    text-align: center;
    color: var(--ctp-mocha-text);
}




</style>

<script lang="ts">
import { METHODS } from 'http';


export default {
    name: 'Contact',
    props: {
        FormData: Object,
    },
    data() {
        return {
            name: 'Click here to add me on Discord',
        }
}
}

</script>