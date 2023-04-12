plugins {
    java
    id("org.springframework.boot") version "3.0.2"
    id("io.spring.dependency-management") version "1.1.0"
}

group = "top.risingentropy"
version = "0.0.1-SNAPSHOT"
java.sourceCompatibility = JavaVersion.VERSION_17

repositories {
    // 依赖使用阿里云 maven 源
    maven {
        setUrl("https://maven.aliyun.com/repository/public/")
    }
    maven {
        setUrl("https://maven.aliyun.com/repository/spring/")
    }
    mavenCentral()
}

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    testImplementation("org.springframework.boot:spring-boot-starter-test")
    implementation("ai.djl.pytorch:pytorch-model-zoo:0.20.0")
//    implementation("org.slf4j:slf4j-reload4j:2.0.6")
    implementation("org.apache.logging.log4j:log4j-core:2.19.0")
    implementation("io.jhdf:jhdf:0.6.9")
// https://mvnrepository.com/artifact/com.alibaba.fastjson2/fastjson2
    implementation("com.alibaba.fastjson2:fastjson2:2.0.23")


}

tasks.withType<Test> {
    useJUnitPlatform()
}
