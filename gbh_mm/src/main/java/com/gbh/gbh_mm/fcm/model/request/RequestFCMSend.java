package com.gbh.gbh_mm.fcm.model.request;

import lombok.Data;

@Data
public class RequestFCMSend {
    private String token;

    private String title;

    private String body;

}
