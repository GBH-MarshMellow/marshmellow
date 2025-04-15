package com.gbh.gbh_mm.s3;

import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.amazonaws.services.s3.model.DeleteObjectRequest;
import com.amazonaws.services.s3.model.ObjectMetadata;
import java.net.URI;
import java.net.URISyntaxException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Component
public class S3Component {
    private final AmazonS3 amazonS3;

    @Value("${aws.s3.bucket-name}")
    private String bucketName;

    @Value("${aws.s3.region}")
    private String region;

    public S3Component(AmazonS3 amazonS3) {
        this.amazonS3 = amazonS3;
    }

    /**
     * MultipartFile을 받아 S3에 업로드 후 파일 URL 반환
     *
     * @param file 업로드할 파일
     * @return 업로드된 파일의 S3 URL
     */
    public String uploadFile(MultipartFile file) {
        // Content-Type에 따라 저장 폴더 결정
        String folder = determineFolder(file.getContentType());

        // 파일 이름 중복을 방지하기 위해 타임스탬프를 접두어로 추가
        String fileName = folder + "/" + System.currentTimeMillis() + "_" + file.getOriginalFilename();

        try {
            ObjectMetadata metadata = new ObjectMetadata();
            metadata.setContentLength(file.getSize());
            metadata.setContentType(file.getContentType());

            // 파일 업로드
            amazonS3.putObject(bucketName, fileName, file.getInputStream(), metadata);

            // 업로드된 파일의 URL 반환
            return amazonS3.getUrl(bucketName, fileName).toString();
        } catch (IOException e) {
            throw new RuntimeException("파일 업로드에 실패했습니다.", e);
        }
    }

    /**
     * 파일의 Content-Type에 따라 저장할 폴더명을 반환
     *
     * @param contentType 업로드 파일의 Content-Type
     * @return S3 버킷 내 폴더 이름
     */
    private String determineFolder(String contentType) {
        if (contentType == null) {
            return "others";
        }
        if (contentType.startsWith("image/")) {
            return "images";
        } else if (contentType.startsWith("audio/")) {
            return "audio";
        } else if (contentType.equals("application/vnd.ms-excel") ||
            contentType.equals("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")) {
            return "excel";
        } else {
            return "others";
        }
    }

    public void deleteFileByUrl(String fileUrl) throws URISyntaxException {
        URI uri = new URI(fileUrl);
        String key = uri.getPath().substring(1);
        amazonS3.deleteObject(new DeleteObjectRequest(bucketName, key));
        System.out.println("파일 삭제 완료: " + key);
    }
}
